# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native shared backward delta launches, independent of graph row admission.

There is no forward operation or final attention gradient in this helper's
contract. Check its complete FP32 delta, optional zeroed scratch and input
canaries; these results do not establish full-chain batch invariance.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM is None, reason="CUDA device required")]


def _compile_dense(operands, delta, dq_accum=None, dq_sem=None, scales=()):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cudnn.sdpa.bwd.kernels.bprop_chain_common import dot_do_o_host, dot_do_o_scaled_host

    scalars = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    }

    def fake(value):
        return None if value is None else make_fake_tensor(scalars[value.dtype], tuple(value.shape), tuple(value.stride()), assumed_align=16)

    arch = f"sm_{_SM}" + ("a" if _SM >= 90 else "")
    options = f"--enable-tvm-ffi --gpu-arch {arch}"
    stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    dim = operands[0].shape[-1]
    args = [fake(value) for value in (*operands, delta)]
    if scales:
        return cute.compile(dot_do_o_scaled_host, *args, *[fake(value) for value in scales], 128, dim, 64, stream, options=options)
    return cute.compile(dot_do_o_host, *args, fake(dq_accum), fake(dq_sem), 128, dim, dim, 64, False, dq_sem is not None, stream, options=options)


def _assert_fresh_bytes(outputs, expected, label):
    # The shared byte comparator counts differences through an Int64
    # reduction. Compare these full snapshots on the host while both the
    # old dO and its replacement are live on the GPU.
    actual = {name: value.cpu() for name, value in snapshot(outputs).items()}
    reference = {name: value.cpu() for name, value in expected.items()}
    assert_bitwise(actual, reference, label)


def _inputs(batch, heads, seqlen, dtype):
    # Exactly representable values with distinct batch/head/row coordinates.
    # The closed-form FP64 oracle covers every output, including large grids.
    amplitude = (torch.arange(batch * heads, dtype=torch.float64).reshape(batch, heads) % 13 - 6) / 8 + 1
    upstream = (torch.arange(seqlen, dtype=torch.float64) % 3 + 1) / 4
    shape = (batch, seqlen, heads, 64)
    o = amplitude.to(device="cuda", dtype=dtype)[:, None, :, None].expand(shape).contiguous()
    do = upstream.to(device="cuda", dtype=dtype)[None, :, None, None].expand(shape).contiguous()
    reference = (64 * amplitude[:, :, None] * upstream[None, None, :]).float().cuda()
    return [o, do], reference


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn], ids=["fp16", "bf16", "fp8"])
@pytest.mark.parametrize(
    "batch,heads,seqlen",
    [(2, 3, 2), (65535, 2, 2), (65536, 2, 2), (2, 65535, 2), (2, 65536, 2), (1, 65536, 129)],
    ids=["small", "batch-edge", "batch-over", "heads-edge", "heads-over", "query-tail"],
)
def test_dense_delta_grid_replay_and_isolation(dtype, batch, heads, seqlen):
    if dtype == torch.float8_e4m3fn and _SM < 89:
        pytest.skip("Native FP8 conversion requires SM89 or newer")
    operands, reference = _inputs(batch, heads, seqlen, dtype)
    rounded = (seqlen + 127) // 128 * 128
    raw = torch.full((batch * heads * rounded + 32,), 12345.0, device="cuda")
    delta = raw[16:-16].view(batch, heads, rounded)
    scales = [torch.tensor([value], device="cuda") for value in (0.25, 0.5)] if dtype == torch.float8_e4m3fn else []
    factor = 0.125 if scales else 1.0
    fn = _compile_dense(operands, delta, scales=scales)

    def launch():
        args = [*operands, delta, *scales] if scales else [*operands, delta, None, None]
        fn(*args, torch.cuda.current_stream().cuda_stream)
        return {"delta": delta}

    def check(multiplier=1.0):
        torch.testing.assert_close(delta[:, :, :seqlen], reference * factor * multiplier, rtol=0, atol=0)
        assert torch.count_nonzero(delta[:, :, seqlen:]) == 0
        assert torch.all(raw[:16] == 12345.0) and torch.all(raw[-16:] == 12345.0)

    expected = assert_replays(launch)
    check()
    # Rebind one input at a time, keeping its old allocation alive through
    # execution. The query-tail case has >1 GiB per half input: cloning both
    # together and making whole-input FP32 temporaries exhausts 16-GiB CI.
    original_o = operands[0].cpu()
    old_o, old_do = operands
    old_raw = raw
    operands[0] = old_o.clone()
    assert operands[0].data_ptr() != old_o.data_ptr()
    _assert_fresh_bytes(launch(), expected, "fresh O binding")
    check()
    assert torch.equal(old_o.cpu().view(torch.uint8), original_o.view(torch.uint8))
    del old_o
    operands[1] = old_do.clone()
    raw = raw.clone()
    delta = raw[16:-16].view(batch, heads, rounded)
    assert operands[1].data_ptr() != old_do.data_ptr() and raw.data_ptr() != old_raw.data_ptr()
    _assert_fresh_bytes(launch(), expected, "fresh dO/output bindings")
    check()

    # A different batch order, duplicated sequence and singleton execution all
    # retain the exact quantized payload and device descales.
    coords = [(batch - 1, heads - 1), (0, 0), (batch - 1, heads - 1), (batch // 2, heads // 2)]
    for chosen in (coords[:1], coords):
        packed = [torch.stack([value[b, :, h] for b, h in chosen]).unsqueeze(2) for value in operands]
        target = torch.empty((len(chosen), 1, rounded), device="cuda")
        packed_fn = _compile_dense(packed, target, scales=scales)
        args = [*packed, target, *scales] if scales else [*packed, target, None, None]
        packed_fn(*args, torch.cuda.current_stream().cuda_stream)
        want = torch.stack([expected["delta"][b, h] for b, h in chosen]).unsqueeze(1)
        assert_bitwise({"delta": target}, {"delta": want}, "reordered/duplicated batch")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    try:
        for multiplier in (-1.0, 0.5):
            # Keep old dO live so a capture with stale pointers cannot pass.
            source, target = old_do.flatten(), operands[1].flatten()
            for start in range(0, source.numel(), 2**20):
                part = source[start : start + 2**20].float().mul_(multiplier).to(dtype)
                target[start : start + 2**20].copy_(part)
            delta.fill_(float("nan"))
            graph.replay()
            check(multiplier)
    finally:
        graph.reset()
    # Inputs are read-only: compare the original O and the changed dO in full.
    assert torch.equal(operands[0].cpu().view(torch.uint8), original_o.view(torch.uint8))
    for start in range(0, source.numel(), 2**20):
        assert torch.equal(target[start : start + 2**20].float(), source[start : start + 2**20].float() * 0.5)
    if (batch, heads) == (2, 3):
        inputs = {"o": operands[0], "do": operands[1], **{f"descale_{i}": value for i, value in enumerate(scales)}}
        assert_golden(f"sdpa_bwd_shared_delta_dense_{str(dtype).split('.')[-1]}", inputs, {"delta": delta})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "batch,heads", [(3, 2), (65535, 2), (65536, 2), (3, 65535), (3, 65536)], ids=["small", "batch-edge", "batch-over", "heads-edge", "heads-over"]
)
def test_thd_origins_delta_grid(dtype, batch, heads):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cudnn.sdpa.bwd.kernels.bprop_chain_common import dot_do_o_thd_origins_host

    lengths = torch.arange(batch, dtype=torch.int32) % 3
    lengths[batch // 2] = 129  # Exercise a live second query tile amid empty sequences.
    prefix = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
    total = int(prefix[-1])
    owners = torch.repeat_interleave(torch.arange(batch), lengths)
    row = torch.arange(total) - prefix[owners]
    amp = ((owners[:, None] + torch.arange(heads)[None, :]) % 13 - 6).double() / 8 + 1
    upstream = (row.double() % 3 + 1) / 4
    reference = (64 * amp * upstream[:, None]).T.float().cuda()
    offsets = torch.stack([prefix[:-1].long() + (torch.arange(batch) + 1) * gap for gap in (1, 2)])
    inputs, positions = [], []
    for port, gap in enumerate((1, 2)):
        value = torch.full((1, total + (batch + 1) * gap, heads, 64), float("nan"), device="cuda", dtype=dtype)
        pos = (offsets[port, owners] + row).cuda()
        live = amp if port == 0 else upstream[:, None].expand(total, heads)
        value[0].index_copy_(0, pos, live.cuda().to(dtype)[:, :, None].expand(total, heads, 64))
        inputs.append(value)
        positions.append(pos)
    meta = torch.zeros(4 * batch + 4, dtype=torch.int32, device="cuda")
    meta[batch : 2 * batch + 1] = prefix.cuda()
    origins = offsets.cuda()
    rounded = (total + 127) // 128 * 128
    raw = torch.full((heads * rounded + 32,), 12345.0, device="cuda")
    delta = raw[16:-16].view(1, heads, rounded)
    scalar = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
    values = [*inputs, delta, meta, origins]
    types = [scalar, scalar, cutlass.Float32, cutlass.Int32, cutlass.Int64]
    fake = [make_fake_tensor(t, tuple(v.shape), tuple(v.stride()), assumed_align=16) for t, v in zip(types, values)]
    del values  # Do not retain old input allocations after staged rebinding.
    arch = f"sm_{_SM}" + ("a" if _SM >= 90 else "")

    @cute.jit
    def host(o: cute.Tensor, do: cute.Tensor, out: cute.Tensor, metadata: cute.Tensor, offsets: cute.Tensor, stream):
        # Production callers supply these immutable declaration facts from
        # their compiled plan, rather than changing them at each launch.
        dot_do_o_thd_origins_host(o, do, out, metadata, offsets, offsets.shape[1], 129, (0, 1), 64, stream)

    fn = cute.compile(
        host,
        *fake,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )

    def launch():
        fn(*inputs, delta, meta, origins, torch.cuda.current_stream().cuda_stream)
        return {"delta": delta}

    def check(multiplier=1.0):
        torch.testing.assert_close(delta[0, :, :total], reference * multiplier, rtol=0, atol=0)
        assert torch.all(delta[:, :, total:] == 12345.0)
        assert torch.all(raw[:16] == 12345.0) and torch.all(raw[-16:] == 12345.0)

    expected = assert_replays(launch)
    check()
    original_o = inputs[0].cpu()
    old_o, old_do = inputs
    inputs[0] = old_o.clone()
    assert inputs[0].data_ptr() != old_o.data_ptr()
    _assert_fresh_bytes(launch(), expected, "THD fresh O storage")
    assert torch.equal(old_o.cpu().view(torch.uint8), original_o.view(torch.uint8))
    del old_o
    inputs[1] = old_do.clone()
    assert inputs[1].data_ptr() != old_do.data_ptr()
    _assert_fresh_bytes(launch(), expected, "THD fresh dO storage")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    try:
        inputs[1].neg_()
        delta.fill_(12345.0)
        graph.replay()
        check(-1)
    finally:
        graph.reset()
    assert torch.equal(inputs[0].cpu().view(torch.uint8), original_o.view(torch.uint8))
    # Poisoned gaps stay poisoned; compare every live dO byte in bounded
    # gathers instead of allocating two full-size gathered inputs at once.
    rows_per_chunk = max(1, 2**20 // (heads * 64))
    for start in range(0, positions[1].numel(), rows_per_chunk):
        pos = positions[1][start : start + rows_per_chunk]
        assert torch.equal(inputs[1][0, pos], -old_do[0, pos])
    if (batch, heads) == (3, 2):
        assert_golden(
            f"sdpa_bwd_shared_delta_thd_{str(dtype).split('.')[-1]}",
            {"o": inputs[0], "do": inputs[1], "meta": meta, "origins": origins},
            {"delta": delta},
        )


def test_flat_delta_initializes_accumulator_and_relay():
    operands, reference = _inputs(1, 65536, 2, torch.bfloat16)
    delta = torch.empty((1, 65536, 128), device="cuda")
    # These optional regions belong to the same preprocess launch. A grid fix
    # that only changes the delta store can leave stale dQ or relay counters.
    accum = torch.full((1, 128, 65536, 64), 5.0, device="cuda")
    relay = torch.full((1, 65536, 1), 7, device="cuda", dtype=torch.int32)
    fn = _compile_dense(operands, delta, accum, relay)
    fn(*operands, delta, accum, relay, torch.cuda.current_stream().cuda_stream)
    torch.testing.assert_close(delta[:, :, :2], reference, rtol=0, atol=0)
    assert torch.count_nonzero(delta[:, :, 2:]) == 0
    # count_nonzero can materialize an Int64 tensor as large as the input.
    # Check every scratch element without a multi-GiB reduction temporary.
    flat_accum = accum.flatten()
    for start in range(0, flat_accum.numel(), 2**20):
        assert torch.count_nonzero(flat_accum[start : start + 2**20]) == 0
    assert torch.count_nonzero(relay) == 0
