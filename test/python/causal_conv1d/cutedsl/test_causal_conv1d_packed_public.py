# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public packed layout, autograd, composition and compiler contracts."""

import json
import subprocess
import sys

import pytest
import torch
from test_causal_conv1d_packed import _dsl, reference

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_packed_dsl():
    _dsl()


TOL = {torch.float16: (0.003, 0.002), torch.bfloat16: (0.03, 0.02), torch.float32: (2e-5, 2e-5)}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("layout", ["token", "channel", "strided"])
def test_public_packed_outputs_history_and_all_gradients(dtype, width, layout):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    lengths, h = [0, 1, 2, 7, 33, 0], 24
    t, s = sum(lengths), len(lengths)
    torch.manual_seed(2718)
    if layout == "channel":
        x = torch.randn(1, h, t, device="cuda", dtype=dtype) * 0.25
        initial = torch.randn(s, h, width - 1, device="cuda", dtype=dtype) * 0.25
    elif layout == "strided":
        x = (torch.randn(1, t * 2, h, device="cuda", dtype=dtype) * 0.25)[:, ::2].transpose(1, 2)
        initial = (torch.randn(s, width - 1, h * 2, device="cuda", dtype=dtype) * 0.25)[:, :, ::2].transpose(1, 2)
    else:
        x = (torch.randn(1, t, h, device="cuda", dtype=dtype) * 0.25).transpose(1, 2)
        initial = (torch.randn(s, width - 1, h, device="cuda", dtype=dtype) * 0.25).transpose(1, 2)
    x.requires_grad_()
    initial.requires_grad_()
    w = (torch.randn(h, width, device="cuda", dtype=dtype) * 0.25).requires_grad_()
    b = (torch.randn(h, device="cuda", dtype=dtype) * 0.25).requires_grad_()
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    final_out = torch.empty(s, width - 1, h * 2, device="cuda", dtype=dtype)[:, :, ::2].transpose(1, 2)
    y, final = causal_conv1d(x, w, b, activation="silu", cu_seqlens=cu, initial_states=initial, return_final_states=True, final_states_out=final_out)
    assert final is final_out
    if layout != "strided":
        assert y.stride() == x.stride()
    xx, ww, bb, ii = (v.double().detach().requires_grad_() for v in (x, w, b, initial))
    ry, rf = reference(xx[0].T, ww, bb, ii.transpose(1, 2), lengths, True)
    ry, rf = ry.T[None], rf.transpose(1, 2)
    atol, rtol = TOL[dtype]
    torch.testing.assert_close(y, ry.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(final, rf.to(dtype), atol=0, rtol=0)
    dy, df = torch.randn_like(y), torch.randn_like(final)
    got = torch.autograd.grad((y, final), (x, w, b, initial), (dy, df))
    want = torch.autograd.grad((ry, rf), (xx, ww, bb, ii), (dy.double(), df.double()))
    for actual, expected in zip(got, want):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_public_chunked_prefill_decode_composition(dtype):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    torch.manual_seed(347)
    lengths = [1, 0, 2, 7, 127]
    x = torch.randn(1, sum(lengths), 24, device="cuda", dtype=dtype).transpose(1, 2) * 0.125
    w = torch.randn(24, 4, device="cuda", dtype=dtype) * 0.125
    initial = torch.randn(1, 3, 24, device="cuda", dtype=dtype).transpose(1, 2) * 0.125
    b = torch.randn(24, device="cuda", dtype=dtype) * 0.125
    results = []
    for chunked in (False, True):
        xx, ww, bb, ii = (v.detach().clone().requires_grad_() for v in (x, w, b, initial))
        state, out, start = ii, [], 0
        for length in lengths if chunked else [sum(lengths)]:
            cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            y, state = causal_conv1d(xx[:, :, start : start + length], ww, bb, activation="silu", cu_seqlens=cu, initial_states=state, return_final_states=True)
            out.append(y)
            start += length
        y = torch.cat(out, dim=2)
        grads = torch.autograd.grad((y, state), (xx, ww, bb, ii), (torch.ones_like(y), torch.ones_like(state)))
        results.append([v.detach().clone() for v in (y, state, *grads)])
    atol, rtol = TOL[dtype]
    for got, want in zip(*results):
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
    torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)


@pytest.mark.parametrize("provided_final", [False, True])
def test_public_fullgraph_aot_and_gradients(provided_final):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    x = torch.randn(1, 11, 24, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    w = torch.randn(24, 4, device="cuda", dtype=torch.bfloat16).requires_grad_()
    b = torch.randn(24, device="cuda", dtype=torch.bfloat16).requires_grad_()
    initial = torch.randn(3, 3, 24, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    cu = torch.tensor([0, 0, 3, 11], device="cuda", dtype=torch.int32)

    def call(x, w, b, initial, cu):
        out = torch.empty_like(initial) if provided_final else None
        return causal_conv1d(x, w, b, activation="silu", cu_seqlens=cu, initial_states=initial, return_final_states=True, final_states_out=out)

    compiled = torch.compile(call, backend="aot_eager", fullgraph=True, dynamic=True)
    expected = call(x, w, b, initial, cu)
    actual = compiled(x, w, b, initial, cu)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    grads = tuple(torch.ones_like(y) for y in expected)
    want = torch.autograd.grad(expected, (x, w, b, initial), grads)
    got = torch.autograd.grad(actual, (x, w, b, initial), grads)
    for g, e in zip(got, want):
        torch.testing.assert_close(g, e, atol=0, rtol=0)


@pytest.mark.parametrize("case", ["first", "last", "decreasing", "negative", "beyond", "empty-invalid"])
def test_public_invalid_offsets_trap_in_an_isolated_process(case):
    import cudnn

    # Reuse the parent's installed package or explicit source overlay. A device
    # trap poisons its CUDA context, so each malformed offset vector gets a child.
    code = """
import json, sys
sys.path[:0] = json.loads(sys.argv[1])
import cudnn
cudnn.__path__[:] = json.loads(sys.argv[2])
import torch
from functools import partial
from cudnn.ops import causal_conv1d as _causal_conv1d
causal_conv1d = partial(_causal_conv1d, backend="se")
case = sys.argv[3]
offsets = {'first': [1, 3], 'last': [0, 2], 'decreasing': [0, 3, 1, 3],
           'negative': [0, -1, 3], 'beyond': [0, 4, 3], 'empty-invalid': [0, 1]}[case]
t = 0 if case == 'empty-invalid' else 3
x = torch.ones(1, t, 24, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
w = torch.ones(24, 4, device='cuda', dtype=torch.bfloat16)
cu = torch.tensor(offsets, device='cuda', dtype=torch.int32)
print('INVALID_PROBE_LAUNCH', case, flush=True)
causal_conv1d(x, w, cu_seqlens=cu, activation='silu')
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(sys.path), json.dumps(list(cudnn.__path__)), case],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    assert result.returncode != 0, result.stdout
    assert "INVALID_PROBE_LAUNCH" in result.stdout, result.stdout
    assert "CUDA error" in result.stdout or "CUDA_ERROR" in result.stdout, result.stdout


def test_compiled_final_output_rechecks_aliases_on_every_call():
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    x = torch.randn(1, 11, 24, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    w = torch.randn(24, 4, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 11], device="cuda", dtype=torch.int32)

    def call(x, w, cu, out):
        return causal_conv1d(x, w, activation="silu", cu_seqlens=cu, return_final_states=True, final_states_out=out)

    compiled = torch.compile(call, backend="aot_eager", fullgraph=True)
    for _ in range(2):
        out = torch.empty(1, 3, 24, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        got = compiled(x, w, cu, out)
        assert got[1] is out
    before = x.clone()
    with pytest.raises(ValueError, match="final_states_out must not share memory with x"):
        compiled(x, w, cu, x[:, :, -3:])
    torch.testing.assert_close(x, before, atol=0, rtol=0)


def test_public_nondefault_stream_graph_replays_changed_empty_boundaries():
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    x = torch.randn(1, 8, 24, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    w = torch.randn(24, 4, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(2, 3, 24, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    cu = torch.tensor([0, 1, 8], device="cuda", dtype=torch.int32)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    def call():
        return causal_conv1d(x, w, activation="silu", cu_seqlens=cu, initial_states=initial, return_final_states=True)

    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        old_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            y, final = call()
        finally:
            torch.cuda.set_sync_debug_mode(old_mode)
    for lengths in ([1, 7], [0, 8], [8, 0]):
        cu.copy_(torch.tensor([0, lengths[0], 8], device="cuda", dtype=torch.int32))
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            graph.replay()
        torch.cuda.current_stream().wait_stream(side)
        expected_y, expected_final = reference(x[0].T.double(), w.double(), None, initial.transpose(1, 2).double(), lengths, True)
        torch.testing.assert_close(y, expected_y.T[None].to(y.dtype), atol=0.03, rtol=0.02)
        torch.testing.assert_close(final, expected_final.transpose(1, 2).to(final.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("lengths", [[1, 1, 1, 1], [2, 0], [0, 1, 7, 513]])
@pytest.mark.parametrize("state_layout", ["SHK", "padded-SKH"])
def test_compact_tokens_with_strided_history(dtype, width, lengths, state_layout):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    torch.manual_seed(617)
    h, s, t = 24, len(lengths), sum(lengths)
    x = (torch.randn(1, t, h, device="cuda", dtype=dtype) * 0.25).transpose(1, 2).requires_grad_()
    w = (torch.randn(h, width, device="cuda", dtype=dtype) * 0.25).requires_grad_()
    b = (torch.randn(h, device="cuda", dtype=dtype) * 0.25).requires_grad_()
    if state_layout == "SHK":
        initial = torch.randn(s, h, width - 1, device="cuda", dtype=dtype) * 0.25
    else:
        initial = (torch.randn(s, width - 1, h * 2, device="cuda", dtype=dtype) * 0.25)[:, :, ::2].transpose(1, 2)
    initial.requires_grad_()
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    final_out = torch.empty(s, h, width - 1, device="cuda", dtype=dtype)
    y, final = causal_conv1d(x, w, b, activation="silu", cu_seqlens=cu, initial_states=initial, return_final_states=True, final_states_out=final_out)
    assert final is final_out and final.is_contiguous()
    xx, ww, bb, ii = (v.double().detach().requires_grad_() for v in (x, w, b, initial))
    ry, rf = reference(xx[0].T, ww, bb, ii.transpose(1, 2), lengths, True)
    ry, rf = ry.T[None], rf.transpose(1, 2)
    atol, rtol = {torch.float16: (0.003, 0.002), torch.bfloat16: (0.03, 0.02), torch.float32: (2e-5, 2e-5)}[dtype]
    torch.testing.assert_close(y, ry.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(final, rf.to(dtype), atol=0, rtol=0)
    dy, df = torch.randn_like(y), torch.randn_like(final)
    got = torch.autograd.grad((y, final), (x, w, b, initial), (dy, df))
    expected = torch.autograd.grad((ry, rf), (xx, ww, bb, ii), (dy.double(), df.double()))
    for actual, want in zip(got, expected):
        torch.testing.assert_close(actual, want.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype,itype,bits",
    [
        (torch.float16, torch.int16, [0, -32768, 0x7E21, 0xFC21 - (1 << 16), 1, 0x7C00, 0x3C00, 0xBC00 - (1 << 16)]),
        (torch.bfloat16, torch.int16, [0, -32768, 0x7FC1, 0xFF81 - (1 << 16), 1, 0x7F80, 0x3F80, 0xBF80 - (1 << 16)]),
        (torch.float32, torch.int32, [0, -(1 << 31), 0x7FC12345, 0xFF812345 - (1 << 32), 1, 0x7F800000, 0x3F800000, 0xBF800000 - (1 << 32)]),
    ],
)
@pytest.mark.parametrize("lengths", [[0, 1, 5, 0], [1, 1, 1, 1]])
def test_public_final_history_preserves_raw_bits(dtype, itype, bits, lengths):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    base = torch.tensor(bits, device="cuda", dtype=itype)
    x = base.repeat(sum(lengths) * 3).reshape(sum(lengths), 24).view(dtype)
    initial = base.repeat(len(lengths) * 9).reshape(len(lengths), 3, 24).view(dtype)
    initial = initial.transpose(1, 2).contiguous().transpose(1, 2)
    w = torch.ones(24, 4, device="cuda", dtype=dtype)
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    final_out = torch.empty(len(lengths), 24, 3, device=x.device, dtype=dtype)
    _, final = causal_conv1d(x.T[None], w, cu_seqlens=cu, initial_states=initial.transpose(1, 2), return_final_states=True, final_states_out=final_out)
    assert final is final_out
    start, states = 0, []
    for seq, length in enumerate(lengths):
        states.append(torch.cat((initial[seq], x[start : start + length]))[-3:])
        start += length
    expected = torch.stack(states)
    assert torch.equal(final.transpose(1, 2).view(itype), expected.view(itype))
