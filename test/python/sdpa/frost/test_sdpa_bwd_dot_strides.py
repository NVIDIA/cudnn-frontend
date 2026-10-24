# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared backward dot helper: physical strides, alignment, rebinding and replay.

These tests call the helper directly; they do not expand graph engine eligibility.
The large declarations reproduce CP8 AllGather's O/dO storage on both Q halves.
"""

import re

import pytest
import torch

from frost_test_utils import _SM, arch_known_to_the_dsl, requires_dsl

pytestmark = [requires_dsl, pytest.mark.skipif(_SM is None or not 100 <= _SM < 130, reason="Shared Blackwell backward helper")]


def _case(dim, dtype, layout, origin=0):
    smoke = dim == 256 and dtype == torch.bfloat16 and layout in ("pitched", "base2", "compact_base2", "compact_o_base2", "compact_do_base2")
    level = pytest.mark.L0 if smoke else pytest.mark.L1
    return pytest.param(dim, dtype, layout, origin, id=f"d{dim}-{str(dtype).split('.')[-1]}-{layout}-{origin}", marks=level)


_CASES = [_case(dim, dtype, layout) for dim, dtype in ((128, torch.bfloat16), (256, torch.float16), (256, torch.bfloat16)) for layout in ("compact", "pitched")]
_CASES += [_case(256, torch.bfloat16, layout) for layout in ("asymmetric", "batch_odd", "row_odd", "head_odd", "base2", "padded")]
_CASES += [_case(256, torch.bfloat16, layout) for layout in ("compact_base2", "compact_o_base2", "compact_do_base2")]
_CASES += [_case(256, torch.bfloat16, "cp8", origin) for origin in (0, 512)]


def _compile(shape, strides, dtype, dim, alignment, arch, dump_dir=None):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cudnn.sdpa.bwd.kernels.bprop_chain_common import dot_do_o_host

    batch, seqlen, heads, _ = shape
    rounded = ((seqlen + 127) // 128) * 128
    scalar = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    alignments = (alignment, alignment) if isinstance(alignment, int) else alignment
    operands = [make_fake_tensor(scalar, shape, stride, assumed_align=align) for stride, align in zip(strides, alignments)]
    delta = make_fake_tensor(cutlass.Float32, (batch, heads, rounded), (heads * rounded, rounded, 1), assumed_align=16)
    options = f"--enable-tvm-ffi --gpu-arch {arch}"
    if dump_dir is not None:
        options += f" --keep-ptx --dump-dir={dump_dir}"
    return cute.compile(
        dot_do_o_host, *operands, delta, None, None, 128, dim, dim, 64, False, False, make_fake_stream(use_tvm_ffi_env_stream=False), options=options
    )


def _reference(operands):
    """Full CPU FP64 row sums, with bounded scratch even for the real CP8 shape."""
    batch, seqlen, heads, _ = operands[0].shape
    result = torch.empty((batch, heads, seqlen), dtype=torch.float64)
    for b in range(batch):
        for begin in range(0, seqlen, 64):
            left, right = [value[b, begin : begin + 64].cpu().double() for value in operands]
            result[b, :, begin : begin + 64] = (left * right).sum(-1).transpose(0, 1)
    return result


@pytest.mark.parametrize("dim,dtype,layout,origin", _CASES)
def test_backward_dot_physical_layout(dim, dtype, layout, origin):
    batch, seqlen, heads = (64, 512, 32) if layout == "cp8" else (2, 257, 4)
    width = 192 if layout == "padded" else dim
    pitch = seqlen if layout.startswith("compact") else 1024 if layout == "cp8" else seqlen + 127
    head_stride = width + int(layout == "head_odd")
    row_stride = heads * head_stride + int(layout == "row_odd")
    strides = [(pitch * row_stride + int(layout == "batch_odd"), row_stride, head_stride, 1)] * 2
    if layout == "asymmetric":
        strides[1] = ((pitch + 128) * row_stride, row_stride, head_stride, 1)
    alignment = {
        "base2": (2, 2),
        "compact_base2": (2, 2),
        "compact_o_base2": (2, 16),
        "compact_do_base2": (16, 2),
    }.get(layout, (16, 16))
    shape = (batch, seqlen, heads, width)
    arch = f"sm_{_SM}a"
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"Installed CuTe DSL cannot compile {arch}")
    torch.manual_seed(358)

    def inputs(extra=0):
        try:
            storage, views = [], []
            for stride, align in zip(strides, alignment):
                offset = 16 + extra + int(align == 2) + origin * row_stride
                span = (batch - 1) * stride[0] + (seqlen - 1) * stride[1] + (heads - 1) * stride[2] + width
                raw = torch.full((offset + span + 16,), 4096.0, dtype=dtype, device="cuda")
                view = raw.as_strided(shape, stride, offset)
                view.copy_(torch.randn(shape, dtype=dtype, device="cuda"))
                storage.append(raw)
                views.append(view)
            return storage, views, [raw.clone() for raw in storage]
        except torch.OutOfMemoryError:
            if layout != "cp8":
                raise
            pytest.skip("Insufficient free memory for complete CP8 physical input and read-only canaries")

    storage, operands, snapshots = inputs()
    rounded = ((seqlen + 127) // 128) * 128
    count = batch * heads * rounded
    raw_delta = torch.full((count + 32,), 12345.0, device="cuda", dtype=torch.float32)
    delta = raw_delta[16 : 16 + count].view(batch, heads, rounded)
    fn = _compile(shape, strides, dtype, dim, alignment, arch)
    stream = torch.cuda.Stream()

    def launch():
        # Both optional tensors remain runtime ABI slots, even when specialized None.
        fn(*operands, delta, None, None, stream.cuda_stream)

    def check(reference):
        torch.cuda.current_stream().wait_stream(stream)
        actual = delta[:, :, :seqlen].cpu().double()
        scale = max(float(reference.abs().max()), 1.0)
        torch.testing.assert_close(actual, reference, rtol=0, atol=2e-5 * scale)
        assert torch.count_nonzero(delta[:, :, seqlen:]) == 0
        assert torch.all(raw_delta[:16] == 12345.0) and torch.all(raw_delta[-16:] == 12345.0)
        assert all(torch.equal(raw, saved) for raw, saved in zip(storage, snapshots))

    reference = _reference(operands)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    check(reference)
    graph = torch.cuda.CUDAGraph()
    try:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            launch()
        for zero in (False, True):
            if zero:
                operands[1].zero_()
            else:
                operands[0].neg_()
                operands[1].mul_(0.5)
            for saved, raw in zip(snapshots, storage):
                saved.copy_(raw)
            reference = _reference(operands)
            delta.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
            check(reference)
    finally:
        graph.reset()
    # Keep old storage alive so the allocator cannot give the rebound frame the same addresses.
    old_storage = storage
    storage, operands, snapshots = inputs(extra=16)
    assert all(new.data_ptr() != old.data_ptr() for new, old in zip(storage, old_storage))
    reference = _reference(operands)
    delta.fill_(float("nan"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    check(reference)


@pytest.mark.L1
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a", "sm_110a", "sm_120a"])
@pytest.mark.parametrize("alignment", [16, 2])
@pytest.mark.parametrize("layout", ["compact", "pitched"])
def test_backward_dot_other_arch_lowering(arch, alignment, layout, tmp_path):
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"Installed CuTe DSL cannot compile {arch}")
    pitch = 257 if layout == "compact" else 384
    _compile((2, 257, 4, 256), [(pitch * 1024, 1024, 256, 1)] * 2, torch.bfloat16, 256, alignment, arch, tmp_path)
    ptx = "\n".join(path.read_text() for path in tmp_path.rglob("*.ptx"))
    assert ptx and "cudnn_kernel_dot_do_o_kernel" in ptx
    vector_load = re.search(r"ld\.global(?:\.[a-z0-9]+)*\.v4\.b32", ptx)
    assert bool(vector_load) == (alignment == 16)
