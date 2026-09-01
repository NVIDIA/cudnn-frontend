# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contract for the exact-shape backward prototype."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.L0


def _load_backward():
    try:
        from cudnn.causal_conv1d_bulk_sm100 import (
            CausalConv1dBulkAutogradPrototype,
            CausalConv1dBulkBwdPrototype,
            backward,
            compile_causal_conv1d_bulk_bwd_prototype,
        )
    except (ImportError, OSError) as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    return (
        backward,
        CausalConv1dBulkAutogradPrototype,
        CausalConv1dBulkBwdPrototype,
        compile_causal_conv1d_bulk_bwd_prototype,
    )


def _support_checked_api(
    monkeypatch,
    *,
    batch=1,
    tokens=257,
    channels=8,
    num_sequences=None,
    schedule="t64",
    capability=(10, 0),
    sm_count=148,
    with_bias=False,
    with_initial_state=False,
    with_d_final_state=False,
    weight_dtype=torch.bfloat16,
):
    backward, _, api_class, _ = _load_backward()
    x = torch.empty(batch, tokens, channels, dtype=torch.bfloat16)
    weight = torch.empty(channels, 4, dtype=weight_dtype)
    dy = torch.empty_like(x)
    bias = torch.zeros(channels, dtype=torch.bfloat16) if with_bias else None
    cu_seqlens = None
    if num_sequences is not None:
        cu_seqlens = torch.empty(num_sequences + 1, dtype=torch.int32)
    state_sequences = batch if num_sequences is None else num_sequences
    initial_state = torch.zeros(state_sequences, channels, 4, dtype=torch.bfloat16) if with_initial_state else None
    d_final_state = torch.zeros(state_sequences, channels, 4, dtype=torch.bfloat16) if with_d_final_state else None
    api = api_class(
        x,
        weight,
        dy,
        sample_cu_seqlens=cu_seqlens,
        schedule=schedule,
        sample_bias=bias,
        sample_initial_state=initial_state,
        sample_d_final_state=d_final_state,
    )
    monkeypatch.setattr(backward, "cutedsl_state", lambda: (True, None))
    monkeypatch.setattr(api, "_require_cuda", lambda desc, name: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(multi_processor_count=sm_count))
    api.check_support()
    return api, x, weight, dy, cu_seqlens, bias, initial_state, d_final_state


def test_auto_schedule_uses_the_measured_t64_t128_cutover():
    backward, *_ = _load_backward()
    assert backward.select_bulk_bwd_schedule(8192) == "t64"
    assert backward.select_bulk_bwd_schedule(16383) == "t64"
    assert backward.select_bulk_bwd_schedule(16384) == "t128"
    assert backward.select_bulk_bwd_schedule(32768) == "t128"


def test_backward_support_accepts_bias_free_fp32_weight(monkeypatch):
    api, _, weight, *_ = _support_checked_api(
        monkeypatch,
        weight_dtype=torch.float32,
    )
    assert api.weight_desc.dtype == torch.float32
    assert weight.dtype == torch.float32


def test_auto_schedule_uses_per_sequence_t_for_dense_and_total_t_for_packed(monkeypatch):
    dense, *_ = _support_checked_api(
        monkeypatch,
        batch=2,
        tokens=8192,
        schedule="auto",
    )
    assert dense.schedule == "t64"

    packed, *_ = _support_checked_api(
        monkeypatch,
        tokens=16384,
        num_sequences=3,
        schedule="auto",
    )
    assert packed.schedule == "t128"

    streaming, *_ = _support_checked_api(
        monkeypatch,
        tokens=8192,
        channels=512,
        schedule="auto",
    )
    assert streaming.schedule == "v2-cpasync"
    assert streaming.kernel_variant == "vec2-cpasync"


@pytest.mark.parametrize("state_kwarg", ("with_initial_state", "with_d_final_state"))
def test_auto_schedule_keeps_state_gradients_on_scalar_kernel(monkeypatch, state_kwarg):
    api, *_ = _support_checked_api(
        monkeypatch,
        tokens=8192,
        channels=512,
        schedule="auto",
        **{state_kwarg: True},
    )
    assert api.schedule == "t64"
    assert api.kernel_variant == "scalar"


def test_vec4_stream_uses_one_sm_scaled_tile_formula(monkeypatch):
    api, *_ = _support_checked_api(
        monkeypatch,
        tokens=16384,
        channels=8192,
        schedule="v4-stream",
    )

    assert api.kernel_variant == "vec4-stream"
    assert api.sm_count == 148
    assert api.tokens_per_cta == 360
    assert api.tiles_per_sequence == 46
    assert api.num_dweight_partials == 46
    assert api.dweight_workspace_bytes == 46 * 8192 * 4 * 4

    short_tail, *_ = _support_checked_api(
        monkeypatch,
        tokens=49,
        channels=512,
        schedule="v4-stream",
    )
    assert short_tail.tokens_per_cta == 25
    assert short_tail.tiles_per_sequence == 2


def test_v2_cpasync_uses_live_sm_count_and_preserves_explicit_vec4(monkeypatch):
    cpasync, *_ = _support_checked_api(
        monkeypatch,
        tokens=16384,
        channels=8192,
        schedule="v2-cpasync",
        sm_count=148,
    )
    assert cpasync.schedule == "v2-cpasync"
    assert cpasync.kernel_variant == "vec2-cpasync"
    assert cpasync.sm_count == 148
    assert cpasync.tokens_per_cta == 915
    assert cpasync.tiles_per_sequence == 18
    assert cpasync.num_dweight_partials == 18
    assert cpasync.dweight_workspace_bytes == 18 * 8192 * 4 * 4

    fewer_sms, *_ = _support_checked_api(
        monkeypatch,
        tokens=16384,
        channels=8192,
        schedule="v2-cpasync",
        capability=(10, 3),
        sm_count=80,
    )
    assert fewer_sms.kernel_variant == "vec2-cpasync"
    assert fewer_sms.tokens_per_cta == 1643
    assert fewer_sms.tiles_per_sequence == 10

    vec4, *_ = _support_checked_api(
        monkeypatch,
        tokens=16384,
        channels=8192,
        schedule="v4-stream",
        sm_count=148,
    )
    assert vec4.kernel_variant == "vec4-stream"
    assert vec4.tokens_per_cta == 360
    assert vec4.tiles_per_sequence == 46


def test_v2_cpasync_planner_is_safe_for_arbitrary_t_d_and_sm(monkeypatch):
    backward, *_ = _load_backward()
    for sequence_length in range(64, 32769):
        for n_channels in (512, 1024, 2048, 4096, 8192, 16384):
            channel_ctas = n_channels // backward._VEC2_CPASYNC_CHANNELS_PER_CTA
            for sm_count in (40, 60, 80, 100, 120, 148, 160):
                token_ctas, tokens_per_cta = backward._plan_vec2_cpasync(sequence_length, n_channels, sm_count)
                last_tile_tokens = sequence_length - (token_ctas - 1) * tokens_per_cta
                assert tokens_per_cta >= 3
                assert (tokens_per_cta - 3) % backward._VEC2_CPASYNC_TOKENS_PER_STAGE == 0
                assert token_ctas == (sequence_length + tokens_per_cta - 1) // tokens_per_cta
                assert (token_ctas - 1) * tokens_per_cta < sequence_length <= token_ctas * tokens_per_cta
                assert last_tile_tokens >= 3
                assert channel_ctas * token_ctas <= sm_count * backward._VEC2_CPASYNC_TARGET_CTAS_PER_SM


def test_v2_cpasync_burst_store_uses_compact_row_major_staging():
    from cudnn.causal_conv1d_bulk_sm100.backward_kernel_vec2_cpasync import (
        _DX_STAGE_STRIDE,
    )

    assert _DX_STAGE_STRIDE == (2, 1)


def test_v2_cpasync_rejects_arch_without_packed_f32x2(monkeypatch):
    with pytest.raises(ValueError, match="requires packed-f32x2 support"):
        _support_checked_api(
            monkeypatch,
            tokens=8192,
            channels=512,
            schedule="v2-cpasync",
            capability=(9, 0),
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"with_bias": True},
        {"with_initial_state": True},
        {"with_d_final_state": True},
        {"num_sequences": 2},
    ),
)
def test_auto_schedule_keeps_cpasync_unsupported_contracts_on_general_kernels(monkeypatch, kwargs):
    api, *_ = _support_checked_api(
        monkeypatch,
        tokens=8192,
        channels=512,
        schedule="auto",
        **kwargs,
    )
    assert api.kernel_variant == "scalar"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch": 2, "tokens": 64, "channels": 512}, "requires B=1"),
        ({"tokens": 64, "channels": 512, "num_sequences": 2}, "dense input only"),
        ({"tokens": 64, "channels": 513}, "D divisible by 512"),
        ({"tokens": 64, "channels": 512, "with_bias": True}, "does not yet support bias"),
        ({"tokens": 64, "channels": 512, "with_initial_state": True}, "does not yet support initial-state"),
        ({"tokens": 64, "channels": 512, "with_d_final_state": True}, "does not yet support initial-state"),
        ({"tokens": 15, "channels": 512}, "requires T>=16"),
    ],
)
def test_vec4_stream_rejects_unimplemented_contracts(monkeypatch, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _support_checked_api(monkeypatch, schedule="v4-stream", **kwargs)


def test_packed_capacity_is_the_tight_shape_only_bound(monkeypatch):
    atomic, *_ = _support_checked_api(
        monkeypatch,
        tokens=257,
        channels=8,
        num_sequences=4,
        schedule="t64",
    )
    assert atomic.packed_tile_capacity == 4 + (257 - 4) // 64 == 7
    assert atomic.packed_tile_map_numel == 28
    assert atomic.packed_tile_map_bytes == 112
    assert atomic.dweight_workspace_bytes == 0
    assert atomic.total_workspace_bytes == 112

    partial, *_ = _support_checked_api(
        monkeypatch,
        tokens=257,
        channels=8,
        num_sequences=4,
        schedule="t64-partial",
    )
    assert partial.num_dweight_partials == 7
    assert partial.dweight_workspace_bytes == 7 * 8 * 4 * 4
    assert partial.total_workspace_bytes == 112 + 7 * 8 * 4 * 4


def test_dense_backward_needs_no_tile_map(monkeypatch):
    api, *_ = _support_checked_api(
        monkeypatch,
        batch=2,
        tokens=65,
        channels=8,
        schedule="t64-partial",
    )
    assert api.tiles_per_sequence == 2
    assert api.num_dweight_partials == 4
    assert api.packed_tile_map_bytes == 0
    assert api.total_workspace_bytes == api.dweight_workspace_bytes


@pytest.mark.parametrize("schedule", ["bad", "", "T64"])
def test_unknown_schedule_fails_at_construction(schedule):
    _, _, api_class, _ = _load_backward()
    x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unknown schedule"):
        api_class(x, torch.zeros(8, 4, dtype=x.dtype), x, schedule=schedule)


def test_packed_requires_one_row_and_no_more_sequences_than_tokens(monkeypatch):
    backward, _, api_class, _ = _load_backward()
    monkeypatch.setattr(backward, "cutedsl_state", lambda: (True, None))

    x = torch.zeros(2, 2, 8, dtype=torch.bfloat16)
    api = api_class(
        x,
        torch.zeros(8, 4, dtype=x.dtype),
        torch.zeros_like(x),
        torch.empty(2, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="Packed X must have B=1"):
        api.check_support()

    x = x[:1]
    api = api_class(
        x,
        torch.zeros(8, 4, dtype=x.dtype),
        torch.zeros_like(x),
        torch.empty(4, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="cannot exceed total_T"):
        api.check_support()


def test_backward_rejects_wrong_packed_metadata_dtype(monkeypatch):
    backward, _, api_class, _ = _load_backward()
    monkeypatch.setattr(backward, "cutedsl_state", lambda: (True, None))
    x = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    api = api_class(
        x,
        torch.zeros(8, 4, dtype=x.dtype),
        torch.zeros_like(x),
        torch.empty(2, dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="cu_seqlens dtype mismatch"):
        api.check_support()


def test_backward_uses_the_shared_architecture_allowlist(monkeypatch):
    with pytest.raises(RuntimeError, match="does not support compute capability"):
        _support_checked_api(monkeypatch, capability=(10, 1))


@pytest.mark.parametrize("case", ["rank", "shape", "dtype", "stride", "alignment"])
def test_backward_rejects_invalid_bias_contract(monkeypatch, case):
    backward, _, api_class, _ = _load_backward()
    x = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, dtype=torch.bfloat16)
    if case == "rank":
        bias = torch.zeros(1, 8, dtype=torch.bfloat16)
    elif case == "shape":
        bias = torch.zeros(7, dtype=torch.bfloat16)
    elif case == "dtype":
        bias = torch.zeros(8, dtype=torch.float32)
    elif case == "stride":
        bias = torch.zeros(8, 2, dtype=torch.bfloat16)[:, 0]
    elif case == "alignment":
        bias = torch.zeros(9, dtype=torch.bfloat16)[1:]
    else:
        raise AssertionError(f"unhandled case {case}")
    api = api_class(x, weight, torch.zeros_like(x), sample_bias=bias)
    monkeypatch.setattr(backward, "cutedsl_state", lambda: (True, None))

    with pytest.raises(ValueError, match="Bias"):
        api.check_support()


def test_backward_rejects_non_tensor_bias():
    _, _, api_class, _ = _load_backward()
    x = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match=r"sample_bias must be a torch\.Tensor or None"):
        api_class(x, torch.zeros(8, 4, dtype=x.dtype), torch.zeros_like(x), sample_bias=object())


def test_bias_and_dbias_presence_match_the_compiled_signature(monkeypatch):
    with_bias, x, weight, dy, _, bias, *_ = _support_checked_api(monkeypatch, tokens=17, with_bias=True)
    with_bias._compiled_kernel = object()
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight, dtype=torch.float32)
    db = torch.empty(weight.shape[0], dtype=torch.float32)

    with pytest.raises(ValueError, match="Bias presence must match"):
        with_bias.execute(x, weight, dy, dx, dw)
    with pytest.raises(ValueError, match="dBias accumulator presence must match"):
        with_bias.execute(x, weight, dy, dx, dw, bias=bias)
    with pytest.raises(TypeError, match=r"dBias accumulator must be a torch\.Tensor"):
        with_bias.execute(x, weight, dy, dx, dw, bias=bias, db_accum=object())
    with pytest.raises(ValueError, match="dBias accumulator must be contiguous FP32"):
        with_bias.execute(x, weight, dy, dx, dw, bias=bias, db_accum=torch.empty(weight.shape[0] + 1, dtype=torch.float32))

    without_bias, *_ = _support_checked_api(monkeypatch, tokens=17)
    without_bias._compiled_kernel = object()
    with pytest.raises(ValueError, match="Bias presence must match"):
        without_bias.execute(x, weight, dy, dx, dw, bias=bias, db_accum=db)


def test_dbias_reuses_dweight_schedule_without_extra_workspace(monkeypatch):
    without_bias, *_ = _support_checked_api(monkeypatch, tokens=257, channels=8, num_sequences=4, schedule="t64-partial")
    with_bias, *_ = _support_checked_api(monkeypatch, tokens=257, channels=8, num_sequences=4, schedule="t64-partial", with_bias=True)

    assert with_bias.dweight_workspace_bytes == without_bias.dweight_workspace_bytes
    assert with_bias.total_workspace_bytes == without_bias.total_workspace_bytes


def test_dbias_accumulator_must_not_overlap_dweight(monkeypatch):
    api, x, weight, dy, _, bias, *_ = _support_checked_api(monkeypatch, tokens=17, with_bias=True)
    api._compiled_kernel = object()
    dw = torch.empty_like(weight, dtype=torch.float32)
    db_alias = dw.view(-1)[: weight.shape[0]]

    with pytest.raises(ValueError, match="dW accumulator and dBias accumulator must not overlap"):
        api.execute(x, weight, dy, torch.empty_like(x), dw, bias=bias, db_accum=db_alias)


def test_state_presence_is_an_exact_compile_signature(monkeypatch):
    api, x, weight, dy, _, bias, initial_state, d_final_state = _support_checked_api(
        monkeypatch,
        tokens=4,
        with_bias=True,
        with_initial_state=True,
        with_d_final_state=True,
    )
    api._compiled_kernel = object()
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight, dtype=torch.float32)
    db = torch.empty(weight.shape[0], dtype=torch.float32)
    d_initial_state = torch.empty_like(initial_state)

    with pytest.raises(ValueError, match="Initial state presence must match"):
        api.execute(
            x,
            weight,
            dy,
            dx,
            dw,
            bias=bias,
            db_accum=db,
            d_final_state=d_final_state,
            d_initial_state=d_initial_state,
        )
    with pytest.raises(ValueError, match="dFinal state presence must match"):
        api.execute(
            x,
            weight,
            dy,
            dx,
            dw,
            bias=bias,
            db_accum=db,
            initial_state=initial_state,
            d_initial_state=d_initial_state,
        )
    with pytest.raises(ValueError, match="dInitial state presence must match"):
        api.execute(
            x,
            weight,
            dy,
            dx,
            dw,
            bias=bias,
            db_accum=db,
            initial_state=initial_state,
            d_final_state=d_final_state,
        )


def test_state_gradient_output_must_not_overlap_inputs(monkeypatch):
    api, x, weight, dy, _, _, initial_state, _ = _support_checked_api(
        monkeypatch,
        tokens=4,
        with_initial_state=True,
    )
    api._compiled_kernel = object()

    with pytest.raises(ValueError, match="Initial state and dInitial state must not overlap"):
        api.execute(
            x,
            weight,
            dy,
            torch.empty_like(x),
            torch.empty_like(weight, dtype=torch.float32),
            initial_state=initial_state,
            d_initial_state=initial_state,
        )


def test_execute_keeps_the_stateless_two_tuple_abi(monkeypatch):
    backward, *_ = _load_backward()
    api, x, weight, dy, *_ = _support_checked_api(
        monkeypatch,
        tokens=17,
        schedule="t64-partial",
    )
    launches = []
    api._compiled_kernel = lambda *args: launches.append(args)
    monkeypatch.setattr(backward, "_as_torch_stream", lambda current_stream, device: object())
    monkeypatch.setattr(backward, "_record_streams", lambda tensors, stream: None)
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight, dtype=torch.float32)
    workspace = torch.empty(api.dweight_workspace_numel, dtype=torch.float32)

    result = api.execute(
        x,
        weight,
        dy,
        dx,
        dw,
        dweight_workspace=workspace,
        current_stream=object(),
    )

    assert len(result) == 2
    assert result[0] is dx
    assert result[1] is dw
    assert len(launches) == 1


def test_execute_rejects_input_output_alias_before_launch(monkeypatch):
    api, x, weight, dy, cu_seqlens, _, *_ = _support_checked_api(
        monkeypatch,
        tokens=17,
        channels=8,
        num_sequences=2,
    )
    api._compiled_kernel = object()
    tile_map = torch.empty(api.packed_tile_map_numel, dtype=torch.int32)
    dw = torch.empty_like(weight, dtype=torch.float32)

    with pytest.raises(ValueError, match="X and dX must not overlap"):
        api.execute(
            x,
            weight,
            dy,
            x,
            dw,
            cu_seqlens=cu_seqlens,
            packed_tile_map=tile_map,
        )
    with pytest.raises(ValueError, match="dY and dX must not overlap"):
        api.execute(
            x,
            weight,
            dy,
            dy,
            dw,
            cu_seqlens=cu_seqlens,
            packed_tile_map=tile_map,
        )


def test_packed_execute_requires_tile_map_even_for_atomic_schedule(monkeypatch):
    api, x, weight, dy, cu_seqlens, _, *_ = _support_checked_api(
        monkeypatch,
        tokens=17,
        channels=8,
        num_sequences=2,
    )
    api._compiled_kernel = object()
    with pytest.raises(ValueError, match="tile-map workspace"):
        api.execute(
            x,
            weight,
            dy,
            torch.empty_like(x),
            torch.empty_like(weight, dtype=torch.float32),
            cu_seqlens=cu_seqlens,
        )
