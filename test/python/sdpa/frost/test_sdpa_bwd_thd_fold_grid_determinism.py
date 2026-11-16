# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared THD dK/dV fold: grid boundaries, physical origins and replay.

This portable backward helper consumes FP32/half partials and emits half gradients.
It has no forward or FP8-output contract. These checks do not qualify the full
attention chain at the same large batch/head declarations.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM is None, reason="CUDA device required")]


def _compile(parts, outputs, meta, origins, batch, max_skv, group, arch=None):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cudnn.sdpa.bwd.kernels.bprop_thd_reduce import dkv_reduce_thd_origins_host

    types = {
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
    }

    def fake(t):
        return make_fake_tensor(types[t.dtype], tuple(t.shape), tuple(t.stride()), assumed_align=16)

    dk_dim, io_dtype = parts[0].shape[3], types[outputs[0].dtype]

    # This wrapper fixes immutable launch facts for both the original and new
    # host, so a parent-source negative control keeps exactly the same test ABI.
    @cute.jit
    def entry(pk, pv, dk, dv, metadata, physical_origins, runtime_skv, stream):
        dkv_reduce_thd_origins_host(pk, pv, dk, dv, metadata, physical_origins, batch, runtime_skv, 0, 1, dk_dim, group, io_dtype, stream)

    arch = arch or f"sm_{_SM}" + ("a" if _SM >= 90 else "")
    compiled = cute.compile(
        entry,
        *[fake(t) for t in (*parts, *outputs, meta, origins)],
        cutlass.Int32(max_skv),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )

    def launch(*args, max_skv_override=None):
        runtime_skv = max_skv if max_skv_override is None else max_skv_override
        return compiled(*args[:-1], cutlass.Int32(runtime_skv), args[-1])

    return launch


def _fixture(batch, heads, last_length, dk_dim, dv_dim, dtype, partial):
    group = 3 if dk_dim == 64 else 1
    lengths = torch.ones(batch, dtype=torch.int64)
    lengths[0] = 0 if batch > 1 else 1
    lengths[-1] = last_length
    prefix = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    total = int(prefix[-1])
    # Only the KV lengths and compact KV prefix are read by this helper.
    meta = torch.cat((lengths, torch.zeros(batch + 1, dtype=torch.int64), prefix)).int().cuda()
    origin_cpu = torch.stack([prefix[:-1] + (torch.arange(batch) + 1) * gap for gap in (1, 2)])
    origins = origin_cpu.cuda()
    token_batch = torch.repeat_interleave(torch.arange(batch), lengths)
    local = torch.arange(total) - torch.repeat_interleave(prefix[:-1], lengths)
    parts, outputs, references, indices = [], [], [], []
    for port, dim in enumerate((dk_dim, dv_dim)):
        # Exact binary fractions: independent FP64 sum has no rounding
        # ambiguity and catches the wrong group, feature, token or head.
        token = (torch.arange(total, dtype=torch.float64) % 11 - 5)[:, None, None, None] / 8
        head = (torch.arange(heads, dtype=torch.float64) % 7)[None, :, None, None] / 16
        member = torch.arange(group, dtype=torch.float64)[None, None, :, None] / 8
        column = (torch.arange(dim, dtype=torch.float64) % 5)[None, None, None, :] / 16
        values = token + head + member + column + port / 8
        parts.append(values.to(dtype=torch.float32 if partial == "fp32" else dtype).reshape(1, total, heads * group, dim).cuda())
        expected = values.sum(2).to(dtype=dtype, device="cuda")
        capacity = total + (port + 1) * (batch + 1) + 1
        output = torch.full((1, capacity, heads, dim), 123.0, dtype=dtype, device="cuda")
        positions = (origin_cpu[port, token_batch] + local).cuda()
        reference = torch.full_like(output, 123.0)
        reference[0, positions] = expected
        outputs.append(output)
        references.append(reference)
        indices.append(positions)
    return parts, outputs, meta, origins, references, indices, lengths, prefix, group


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("partial", ["fp32", "io"], ids=["fp32-partials", "half-partials"])
@pytest.mark.parametrize("dk_dim,dv_dim", [(64, 128), (128, 64)], ids=["dk64-dv128-gqa3", "dk128-dv64-mha"])
@pytest.mark.parametrize(
    "batch,heads,last_length",
    [(3, 2, 35), (65535, 1, 1), (65536, 1, 1), (2, 65535, 1), (2, 65536, 1), (65536, 1, 35)],
    ids=["small", "batch-edge", "batch-over", "heads-edge", "heads-over", "live-second-block"],
)
def test_thd_fold_grid_replay_and_isolation(dtype, partial, dk_dim, dv_dim, batch, heads, last_length):
    parts, outputs, meta, origins, refs, indices, lengths, prefix, group = _fixture(batch, heads, last_length, dk_dim, dv_dim, dtype, partial)
    fn = _compile(parts, outputs, meta, origins, batch, last_length, group)
    frozen = snapshot({"pk": parts[0], "pv": parts[1], "meta": meta, "origins": origins})

    def launch():
        fn(*parts, *outputs, meta, origins, torch.cuda.current_stream().cuda_stream)
        return {"dk": outputs[0], "dv": outputs[1]}

    def check(factor=1):
        for out, ref, positions in zip(outputs, refs, indices):
            want = ref.clone()
            want[0, positions] *= factor
            torch.testing.assert_close(out, want, rtol=0, atol=0)

    result = assert_replays(launch)
    check()
    assert_bitwise(snapshot({"pk": parts[0], "pv": parts[1], "meta": meta, "origins": origins}), frozen, "input preservation")

    # Fresh storage, including all physical origins and metadata, must bind to
    # the retained callable. Keep the old allocations alive while rebinding.
    old = parts, outputs, meta, origins
    parts = [t.clone() for t in parts]
    outputs = [torch.full_like(t, 123.0) for t in outputs]
    meta, origins = meta.clone(), origins.clone()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(parts + outputs, old[0] + old[1]))
    assert_bitwise(snapshot(launch()), result, "fresh bindings")

    # Sequence alone, reordered and duplicated. Remap compact prefixes and
    # each output's independent origin while preserving all partial heads.
    for chosen in ([batch - 1], [batch - 1, 0, batch - 1]):
        lens = lengths[chosen]
        pref = torch.cat((torch.zeros(1, dtype=torch.int64), lens.cumsum(0)))
        packed_parts = [torch.cat([p[:, prefix[b] : prefix[b + 1]] for b in chosen], dim=1) for p in parts]
        packed_meta = torch.cat((lens, torch.zeros(len(chosen) + 1, dtype=torch.int64), pref)).int().cuda()
        packed_origins = torch.stack((pref[:-1] + 1, pref[:-1] + 2)).cuda()
        packed_outputs = [torch.full((1, int(pref[-1]) + 4, heads, d), 123.0, device="cuda", dtype=dtype) for d in (dk_dim, dv_dim)]
        packed_refs = [t.clone() for t in packed_outputs]
        for port in range(2):
            for n, b in enumerate(chosen):
                count = int(lengths[b])
                original_start = int(origins[port, b])
                packed_start = int(pref[n]) + port + 1
                packed_refs[port][:, packed_start : packed_start + count] = outputs[port][:, original_start : original_start + count]
        packed_fn = _compile(packed_parts, packed_outputs, packed_meta, packed_origins, len(chosen), last_length, group)
        packed_fn(*packed_parts, *packed_outputs, packed_meta, packed_origins, torch.cuda.current_stream().cuda_stream)
        assert_bitwise(snapshot(dict(zip(("dk", "dv"), packed_outputs))), snapshot(dict(zip(("dk", "dv"), packed_refs))), "packed canaries")
        for port in range(2):
            for n, b in enumerate(chosen):
                count = int(lengths[b])
                original_start = int(origins[port, b])
                packed_start = int(pref[n]) + port + 1
                assert_bitwise(
                    snapshot({"grad": packed_outputs[port][:, packed_start : packed_start + count]}),
                    snapshot({"grad": outputs[port][:, original_start : original_start + count]}),
                    "batch remapping",
                )

    if batch > 2:
        target = [out[:, int(origins[port, -1]) : int(origins[port, -1]) + last_length].clone() for port, out in enumerate(outputs)]
        for p in parts:
            p[:, prefix[1] : prefix[2]].add_(2)
        launch()
        for port, out in enumerate(outputs):
            start = int(origins[port, -1])
            assert_bitwise(snapshot({"target": out[:, start : start + last_length]}), snapshot({"target": target[port]}), "changed neighbor")
        for p in parts:
            p[:, prefix[1] : prefix[2]].sub_(2)
        assert_bitwise(snapshot(launch()), result, "restored neighbor")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    try:
        for part in parts:
            part.neg_()
        for out in outputs:
            out.fill_(123.0)
        graph.replay()
        check(-1)
        negative = snapshot(launch())
        assert_bitwise(snapshot(launch()), negative, "changed-input replay")
        for part in parts:
            part.neg_()
        for out in outputs:
            out.fill_(123.0)
        graph.replay()
        check()
        assert_bitwise(snapshot(launch()), result, "restored replay")
    finally:
        graph.reset()
    if batch == 3:
        assert_golden(f"sdpa_bwd_thd_fold_grid_{str(dtype).split('.')[-1]}_{partial}_dk{dk_dim}_dv{dv_dim}", frozen, result)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "batch,heads,length,dk_dim,dv_dim",
    [(65536, 32, 4097, 512, 256), (65536, 32, 8193, 512, 256), (65536, 1, 2097184, 64, 64), (1, 1, 2097184, 64, 64)],
    ids=["above-grid-x-limit", "above-uint32-grid", "runtime-long-sequence", "ordinary-grid-stride"],
)
def test_thd_fold_complete_grid_product_keeps_last_live_owner(dtype, batch, heads, length, dk_dim, dv_dim):
    """A compact valid allocation has live work beyond the old narrowed grid.

    Most declared sequences are empty. The first sequence crosses the runtime
    block limit and the last sequence/head is a canary for lost ownership.
    No full large-batch attention-chain claim follows from this helper test.
    """
    lengths = torch.zeros(batch, dtype=torch.int64)
    lengths[0] = length
    if batch > 1:
        lengths[-1] = 1
    prefix = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    total = int(prefix[-1])
    meta = torch.cat((lengths, torch.zeros(batch + 1, dtype=torch.int64), prefix)).int().cuda()
    origin_cpu = torch.zeros((2, batch), dtype=torch.int64)
    origin_cpu[0, 0], origin_cpu[1, 0] = 3, 2
    if batch > 1:
        origin_cpu[0, -1], origin_cpu[1, -1] = 0, total + 4
    origins = origin_cpu.cuda()
    parts = [torch.ones((1, total, heads, dim), device="cuda", dtype=torch.float32) for dim in (dk_dim, dv_dim)]
    outputs = [torch.full((1, total + 7, heads, dim), 123.0, device="cuda", dtype=dtype) for dim in (dk_dim, dv_dim)]
    fn = _compile(parts, outputs, meta, origins, batch, length, 1)

    def launch():
        fn(*parts, *outputs, meta, origins, torch.cuda.current_stream().cuda_stream)
        return dict(zip(("dk", "dv"), outputs))

    def check(value):
        for port, output in enumerate(outputs):
            ref = torch.full_like(output, 123.0)
            for sequence in ([0, batch - 1] if batch > 1 else [0]):
                begin, count = int(origin_cpu[port, sequence]), int(lengths[sequence])
                ref[:, begin : begin + count] = value
            torch.testing.assert_close(output, ref, rtol=0, atol=0)

    launch()
    torch.cuda.synchronize()
    check(1)
    # Each complete-grid output can contain 134M elements. The shared byte
    # comparator's Int64 reduction would add a multi-GiB GPU temporary while
    # the inputs and replay snapshots are live; compare every byte on CPU.
    before = {name: value.cpu() for name, value in snapshot(launch()).items()}
    for output in outputs:
        output.fill_(123.0)
    replay = {name: value.cpu() for name, value in snapshot(launch()).items()}
    assert_bitwise(replay, before, "complete-grid replay")
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured):
            launch()
        for part in parts:
            part.fill_(2)
        for output in outputs:
            output.fill_(123.0)
        captured.replay()
        torch.cuda.synchronize()
        check(2)
    finally:
        captured.reset()
