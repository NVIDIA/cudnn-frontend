# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L0 GPU correctness for the SM100 bulk causal-convolution forward API."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from cuda.bindings import driver as cuda

from fe_api.causal_conv1d_bulk.reference import (
    causal_conv1d_bulk_reference,
    causal_conv1d_update_reference,
)


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


pytestmark = [
    pytest.mark.L0,
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
    pytest.mark.skipif(not _is_sm100(), reason="requires exactly SM100 (compute capability 10.0)"),
]


def _load_wrapper():
    try:
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
        from cudnn.causal_conv1d_bulk_sm100 import causal_conv1d_bulk_fwd_wrapper_sm100
    except (ImportError, OSError) as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("causal_conv1d_bulk_sm100 requires nvidia-cutlass-dsl>=4.7.0")
    return causal_conv1d_bulk_fwd_wrapper_sm100


def _load_class():
    try:
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
        from cudnn.causal_conv1d_bulk_sm100 import CausalConv1dBulkFwdSm100
    except (ImportError, OSError) as error:
        pytest.skip(f"CuTe DSL dependencies unavailable: {error}")
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("causal_conv1d_bulk_sm100 requires nvidia-cutlass-dsl>=4.7.0")
    return CausalConv1dBulkFwdSm100


def _assert_state_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@torch.no_grad()
def test_dense_channel_tail_without_state_has_stable_outputs():
    from cudnn.api_base import TupleDict

    wrapper = _load_wrapper()
    torch.manual_seed(101)
    x = torch.randn(2, 7, 259, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(259, 4, device="cuda", dtype=torch.bfloat16)
    expected, _ = causal_conv1d_bulk_reference(x, weight)

    result = wrapper(x, weight, output_final_state=False)

    assert isinstance(result, TupleDict)
    assert list(result.keys()) == ["output_tensor", "final_state_tensor"]
    assert result[0] is result["output_tensor"]
    assert result[1] is result["final_state_tensor"]
    torch.testing.assert_close(result["output_tensor"], expected, atol=3e-2, rtol=3e-2)
    assert result["final_state_tensor"].shape == (0,)
    assert result["final_state_tensor"].dtype == torch.bfloat16
    assert result["final_state_tensor"].device == x.device
    assert result["final_state_tensor"].is_contiguous()

    # T is intentionally absent from the wrapper cache key: one symbolic
    # compile must safely serve a different runtime token extent.
    x2 = torch.randn(2, 11, 259, device="cuda", dtype=torch.bfloat16)
    expected2, _ = causal_conv1d_bulk_reference(x2, weight)
    result2 = wrapper(x2, weight, output_final_state=False)
    torch.testing.assert_close(result2["output_tensor"], expected2, atol=3e-2, rtol=3e-2)

    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        wrapper(x.detach().requires_grad_(True), weight)


@torch.no_grad()
def test_dense_class_nonzero_state_final_state_and_decode_recurrence():
    api_class = _load_class()
    torch.manual_seed(102)
    batch, tokens, channels = 2, 3, 257
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(batch, channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state_before = initial_state.clone()
    expected, expected_final = causal_conv1d_bulk_reference(x, weight, initial_state=initial_state)

    output = torch.empty_like(x)
    final_state = torch.empty_like(initial_state)
    api = api_class(
        sample_x=x,
        sample_weight=weight,
        sample_output=output,
        sample_initial_state=initial_state,
        sample_final_state=final_state,
    )
    assert api.check_support()
    api.compile()
    result = api.execute(
        x_tensor=x,
        weight_tensor=weight,
        output_tensor=output,
        initial_state_tensor=initial_state,
        final_state_tensor=final_state,
    )

    assert list(result.keys()) == ["output_tensor", "final_state_tensor"]
    assert result["output_tensor"] is output
    assert result["final_state_tensor"] is final_state
    torch.testing.assert_close(result["output_tensor"], expected, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(result["final_state_tensor"], expected_final)
    _assert_state_bits_equal(initial_state, initial_state_before)

    # The full-width state needs no repacking before the next decode token.
    next_x = torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16)
    expected_next, expected_after_decode = causal_conv1d_update_reference(next_x, expected_final, weight)
    actual_next, actual_after_decode = causal_conv1d_update_reference(next_x, result["final_state_tensor"], weight)
    torch.testing.assert_close(actual_next, expected_next, atol=0, rtol=0)
    _assert_state_bits_equal(actual_after_decode, expected_after_decode)

    # Exercise the native decode implementation as well once that sibling API
    # is present in the checkout (the bulk branch can precede it in CI).
    try:
        from cudnn.causal_conv1d_update_sm100 import causal_conv1d_update
    except (ImportError, OSError):
        causal_conv1d_update = None
    if causal_conv1d_update is not None:
        decode_state = result["final_state_tensor"].clone()
        decode_output = causal_conv1d_update(next_x, decode_state, weight)
        torch.testing.assert_close(decode_output, expected_next, atol=3e-2, rtol=3e-2)
        _assert_state_bits_equal(decode_state, expected_after_decode)

    # The preallocated class API revalidates outputs and aliases at execute,
    # rather than trusting the descriptors used for compilation.
    with pytest.raises(ValueError, match="Output T must match X"):
        api.execute(
            x,
            weight,
            torch.empty(batch, tokens + 1, channels, device="cuda", dtype=x.dtype),
            initial_state_tensor=initial_state,
            final_state_tensor=final_state,
        )
    with pytest.raises(TypeError, match="Output dtype mismatch"):
        api.execute(x, weight, output.float(), initial_state_tensor=initial_state, final_state_tensor=final_state)
    with pytest.raises(ValueError, match="Output must be.*contiguous"):
        api.execute(
            x,
            weight,
            torch.empty(batch, tokens, channels * 2, device="cuda", dtype=x.dtype)[..., ::2],
            initial_state_tensor=initial_state,
            final_state_tensor=final_state,
        )
    with pytest.raises(ValueError, match="Final state presence must match"):
        api.execute(x, weight, output, initial_state_tensor=initial_state)
    with pytest.raises(ValueError, match="Final state stride mismatch"):
        api.execute(
            x,
            weight,
            output,
            initial_state_tensor=initial_state,
            final_state_tensor=torch.empty(batch, channels, 8, device="cuda", dtype=x.dtype)[..., ::2],
        )
    with pytest.raises(ValueError, match="Output must not overlap"):
        api.execute(x, weight, x, initial_state_tensor=initial_state, final_state_tensor=final_state)
    with pytest.raises(ValueError, match="Final state must not overlap"):
        api.execute(x, weight, output, initial_state_tensor=initial_state, final_state_tensor=initial_state)
    dlpack_output_alias = torch.from_dlpack(x)
    assert dlpack_output_alias.data_ptr() == x.data_ptr()
    with pytest.raises(ValueError, match="Output must not overlap"):
        api.execute(x, weight, dlpack_output_alias, initial_state_tensor=initial_state, final_state_tensor=final_state)
    dlpack_final_alias = torch.from_dlpack(initial_state)
    assert dlpack_final_alias.data_ptr() == initial_state.data_ptr()
    with pytest.raises(ValueError, match="Final state must not overlap"):
        api.execute(x, weight, output, initial_state_tensor=initial_state, final_state_tensor=dlpack_final_alias)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        api.execute(x.detach().requires_grad_(True), weight, output, initial_state_tensor=initial_state, final_state_tensor=final_state)


@torch.no_grad()
def test_vec8_dense_initial_and_final_state_match_reference():
    wrapper = _load_wrapper()
    torch.manual_seed(104)
    batch, tokens, channels = 2, 9, 264
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(batch, channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state_before = initial_state.clone()
    expected, expected_final = causal_conv1d_bulk_reference(x, weight, initial_state=initial_state)

    result = wrapper(
        x,
        weight,
        initial_state_tensor=initial_state,
        output_final_state=True,
    )

    torch.testing.assert_close(result["output_tensor"], expected, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(result["final_state_tensor"], expected_final)
    _assert_state_bits_equal(initial_state, initial_state_before)

    # The vec8 specialization must retain the same symbolic-T cache contract
    # as the scalar fallback, including execution on a caller-owned stream.
    x2 = torch.randn(batch, 13, channels, device="cuda", dtype=torch.bfloat16)
    expected2, expected_final2 = causal_conv1d_bulk_reference(x2, weight, initial_state=initial_state)
    launch_stream = torch.cuda.Stream()
    torch.cuda.current_stream().synchronize()
    result2 = wrapper(
        x2,
        weight,
        initial_state_tensor=initial_state,
        output_final_state=True,
        current_stream=cuda.CUstream(launch_stream.cuda_stream),
    )
    launch_stream.synchronize()
    torch.testing.assert_close(result2["output_tensor"], expected2, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(result2["final_state_tensor"], expected_final2)


@torch.no_grad()
@pytest.mark.parametrize(
    "use_initial_state,output_final_state",
    [(False, True), (True, False)],
)
def test_optional_state_presence_compile_specializations(use_initial_state, output_final_state):
    wrapper = _load_wrapper()
    torch.manual_seed(107 + int(use_initial_state))
    batch, tokens, channels = 2, 5, 264
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(batch, channels, 4, device="cuda", dtype=torch.bfloat16) if use_initial_state else None
    initial_state_before = initial_state.clone() if initial_state is not None else None
    expected, expected_final = causal_conv1d_bulk_reference(x, weight, initial_state=initial_state)

    result = wrapper(
        x,
        weight,
        initial_state_tensor=initial_state,
        output_final_state=output_final_state,
    )

    torch.testing.assert_close(result["output_tensor"], expected, atol=3e-2, rtol=3e-2)
    if output_final_state:
        _assert_state_bits_equal(result["final_state_tensor"], expected_final)
    else:
        assert result["final_state_tensor"].shape == (0,)
    if initial_state is not None:
        _assert_state_bits_equal(initial_state, initial_state_before)


@torch.no_grad()
def test_packed_unequal_sequences_do_not_bleed_across_boundaries():
    wrapper = _load_wrapper()
    torch.manual_seed(103)
    lengths = (17, 2, 14)  # Crosses scalar token tiles and includes T < width.
    channels = 263
    total_tokens = sum(lengths)
    x = torch.randn(1, total_tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor((0, 17, 19, 33), device="cuda", dtype=torch.int32)
    initial_state = torch.randn(len(lengths), channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state_before = initial_state.clone()
    expected, expected_final = causal_conv1d_bulk_reference(
        x,
        weight,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )

    result = wrapper(
        x,
        weight,
        cu_seqlens_tensor=cu_seqlens,
        initial_state_tensor=initial_state,
        output_final_state=True,
    )

    torch.testing.assert_close(result["output_tensor"], expected, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(result["final_state_tensor"], expected_final)
    _assert_state_bits_equal(initial_state, initial_state_before)

    # The first output of each sequence must depend on its own initial state,
    # never on the token immediately before the packed boundary.
    for sequence, begin in enumerate(cu_seqlens[:-1].tolist()):
        one_token, _ = causal_conv1d_bulk_reference(
            x[:, begin : begin + 1],
            weight,
            initial_state=initial_state[sequence : sequence + 1],
        )
        torch.testing.assert_close(result["output_tensor"][:, begin : begin + 1], one_token, atol=3e-2, rtol=3e-2)

    # Packed N/D/state presence form the compile signature, while total_T and
    # the boundary values remain runtime inputs. Reuse this scalar compiled
    # object at a second legal T with different cross-tile boundaries.
    lengths2 = (8, 1, 12)
    x2 = torch.randn(1, sum(lengths2), channels, device="cuda", dtype=torch.bfloat16)
    cu_seqlens2 = torch.tensor((0, 8, 9, 21), device="cuda", dtype=torch.int32)
    expected2, expected_final2 = causal_conv1d_bulk_reference(
        x2,
        weight,
        cu_seqlens=cu_seqlens2,
        initial_state=initial_state,
    )
    result2 = wrapper(
        x2,
        weight,
        cu_seqlens_tensor=cu_seqlens2,
        initial_state_tensor=initial_state,
        output_final_state=True,
    )
    torch.testing.assert_close(result2["output_tensor"], expected2, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(result2["final_state_tensor"], expected_final2)


@torch.no_grad()
@pytest.mark.parametrize("pass_explicit_handle", [False, True])
def test_side_stream_launch_records_temporary_operand_lifetimes(pass_explicit_handle):
    # A raw CuTe launch is invisible to PyTorch's allocator. Delay a side-stream
    # launch, drop every read operand, then pressure the matching allocation
    # buckets on the default stream. This covers both an explicit handle and
    # current_stream=None while a non-default torch stream is current.
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")

    wrapper = _load_wrapper()
    torch.manual_seed(106)
    batch, tokens, channels = 2, 13, 264
    warm_x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
    warm_weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    warm_state = torch.randn(batch, channels, 4, device="cuda", dtype=torch.bfloat16)
    wrapper(warm_x, warm_weight, initial_state_tensor=warm_state, output_final_state=True)
    torch.cuda.synchronize()

    ephemeral_x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
    ephemeral_weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    ephemeral_state = torch.randn(batch, channels, 4, device="cuda", dtype=torch.bfloat16)
    expected, expected_final = causal_conv1d_bulk_reference(
        ephemeral_x,
        ephemeral_weight,
        initial_state=ephemeral_state,
    )
    torch.cuda.current_stream().synchronize()

    delayed_stream = torch.cuda.Stream()
    with torch.cuda.stream(delayed_stream):
        torch.cuda._sleep(100_000_000)
        delayed_result = wrapper(
            ephemeral_x,
            ephemeral_weight,
            initial_state_tensor=ephemeral_state,
            output_final_state=True,
            current_stream=cuda.CUstream(delayed_stream.cuda_stream) if pass_explicit_handle else None,
        )

    del ephemeral_x, ephemeral_weight, ephemeral_state
    poison = []
    for _ in range(8):
        poison.extend(
            (
                torch.full((batch, tokens, channels), float("nan"), device="cuda", dtype=torch.bfloat16),
                torch.full((channels, 4), float("nan"), device="cuda", dtype=torch.bfloat16),
                torch.full((batch, channels, 4), float("nan"), device="cuda", dtype=torch.bfloat16),
            )
        )
    delayed_stream.synchronize()
    torch.testing.assert_close(delayed_result["output_tensor"], expected, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(delayed_result["final_state_tensor"], expected_final)


@torch.no_grad()
def test_vec8_packed_short_sequences_match_reference_and_final_state_bits():
    wrapper = _load_wrapper()
    torch.manual_seed(105)
    lengths = (1, 2, 3, 4, 5)
    channels = 264
    x = torch.randn(1, sum(lengths), channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor((0, 1, 3, 6, 10, 15), device="cuda", dtype=torch.int32)
    initial_state = torch.randn(len(lengths), channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state_before = initial_state.clone()
    expected, expected_final = causal_conv1d_bulk_reference(
        x,
        weight,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )

    result = wrapper(
        x,
        weight,
        cu_seqlens_tensor=cu_seqlens,
        initial_state_tensor=initial_state,
        output_final_state=True,
    )

    torch.testing.assert_close(result["output_tensor"], expected, atol=3e-2, rtol=3e-2)
    _assert_state_bits_equal(result["final_state_tensor"], expected_final)
    _assert_state_bits_equal(initial_state, initial_state_before)

    # N is part of the cache key but T is symbolic. Rebinding this cached packed
    # signature to fewer tokens than positive sequences must fail on the host,
    # before the device-side metadata validator can trap.
    too_short_x = torch.randn(1, 3, channels, device="cuda", dtype=torch.bfloat16)
    too_short_cu = torch.tensor((0, 1, 2, 3, 3, 3), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="cannot exceed total_T"):
        wrapper(
            too_short_x,
            weight,
            cu_seqlens_tensor=too_short_cu,
            initial_state_tensor=initial_state,
            output_final_state=True,
        )


@pytest.mark.parametrize(
    "case",
    [
        "x-rank",
        "weight-shape",
        "x-dtype",
        "weight-dtype",
        "x-stride",
        "weight-stride",
        "packed-batch",
        "packed-too-many-sequences",
        "cu-too-short",
        "cu-dtype",
        "cu-stride",
        "state-shape",
        "state-dtype",
        "state-stride",
        "cu-device",
    ],
)
def test_invalid_public_contract_is_rejected_before_launch(case):
    wrapper = _load_wrapper()
    x = torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(8, 4, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = None
    initial_state = None

    if case == "x-rank":
        x = x.squeeze(0)
    elif case == "weight-shape":
        weight = weight[:, :3]
    elif case == "x-dtype":
        x = x.float()
    elif case == "weight-dtype":
        weight = weight.float()
    elif case == "x-stride":
        x = torch.empty(1, 8, 4, device="cuda", dtype=x.dtype).transpose(1, 2)
    elif case == "weight-stride":
        weight = torch.empty(8, 8, device="cuda", dtype=weight.dtype)[:, ::2]
    elif case == "packed-batch":
        x = x.expand(2, -1, -1).contiguous()
        cu_seqlens = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    elif case == "packed-too-many-sequences":
        cu_seqlens = torch.arange(6, device="cuda", dtype=torch.int32)
    elif case == "cu-too-short":
        cu_seqlens = torch.empty(0, device="cuda", dtype=torch.int32)
    elif case == "cu-dtype":
        cu_seqlens = torch.tensor([0, 4], device="cuda", dtype=torch.int64)
    elif case == "cu-stride":
        cu_seqlens = torch.tensor([0, -1, 4, -1], device="cuda", dtype=torch.int32)[::2]
    elif case == "state-shape":
        initial_state = torch.zeros(2, 8, 4, device="cuda", dtype=x.dtype)
    elif case == "state-dtype":
        initial_state = torch.zeros(1, 8, 4, device="cuda", dtype=torch.float32)
    elif case == "state-stride":
        initial_state = torch.empty(1, 8, 8, device="cuda", dtype=x.dtype)[:, :, ::2]
    elif case == "cu-device":
        cu_seqlens = torch.tensor([0, 4], device="cpu", dtype=torch.int32)
    else:
        raise AssertionError(f"unhandled case {case}")

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        wrapper(
            x,
            weight,
            cu_seqlens_tensor=cu_seqlens,
            initial_state_tensor=initial_state,
            output_final_state=True,
        )


_DEVICE_ASSERT_WORKER = r"""
import os
import sys

import torch
import cudnn

case = sys.argv[1]
source_cudnn = sys.argv[2]
cudnn.__path__.insert(0, source_cudnn)

from cudnn.causal_conv1d_bulk_sm100 import CausalConv1dBulkFwdSm100

torch.manual_seed(19)
n_channels = 257
x = torch.randn(1, 6, n_channels, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(n_channels, 4, device="cuda", dtype=torch.bfloat16)
output = torch.empty_like(x)
valid_cu = torch.tensor([0, 2, 4, 6], device="cuda", dtype=torch.int32)
api = CausalConv1dBulkFwdSm100(x, weight, output, sample_cu_seqlens=valid_cu)
api.check_support()
api.compile()
api.execute(x, weight, output, cu_seqlens_tensor=valid_cu)
torch.cuda.synchronize()

invalid_cu = {
    "start_nonzero": [1, 2, 4, 6],
    "end_mismatch": [0, 2, 4, 5],
    "non_increasing": [0, 4, 3, 6],
}[case]
invalid_cu = torch.tensor(invalid_cu, device="cuda", dtype=torch.int32)

try:
    api.execute(x, weight, output, cu_seqlens_tensor=invalid_cu)
    torch.cuda.synchronize()
except Exception as error:
    print(f"EXPECTED_DEVICE_FAILURE:{case}:{type(error).__name__}:{error}", flush=True)
    os._exit(0)

print(f"MISSING_DEVICE_FAILURE:{case}", flush=True)
os._exit(9)
"""


@pytest.mark.parametrize("case", ["start_nonzero", "end_mismatch", "non_increasing"])
def test_invalid_cu_seqlens_fail_closed_in_fresh_process(case):
    # A PTX trap poisons its CUDA context, so isolate each metadata class and
    # first prove that the same compiled object accepts valid boundaries.
    source_cudnn = Path(__file__).resolve().parents[4] / "python" / "cudnn"
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _DEVICE_ASSERT_WORKER, case, str(source_cudnn)],
        cwd=Path(__file__).resolve().parents[4],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    diagnostics = result.stdout + result.stderr
    assert result.returncode == 0, diagnostics
    assert f"EXPECTED_DEVICE_FAILURE:{case}:" in diagnostics, diagnostics
