# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native A-mode candidates for small THD decode graphs (#1084).

cuDNN 9.27 offers a THD single-query candidate that fails NVRTC compilation,
including aligned KV and Stats-disabled graphs. The retained native alternative
is valid. Keep candidate compilation separate from default/ALL fallback, and
check dense, prefill and backward controls with the same packed declarations.
"""

import math

import cudnn
import pytest
import torch

from cudnn.engines import is_backend_engine

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(cudnn.backend_version() < 92800, reason="issue1084: exercise native decode candidates on cuDNN 9.28+"),
]


def _case(handle, dtype, sq, skv, stats, thd, offset64, backward):
    b, hq, hk, d = 2, 8, 2, 128
    qlens, klens = (sq, max(1, sq - 1)), (skv, skv - 3)
    torch.manual_seed(1084)
    values = {
        "q": torch.randn(b, sq, hq, d, device="cuda", dtype=dtype) * 0.25,
        "k": torch.randn(b, skv, hk, d, device="cuda", dtype=dtype) * 0.25,
        "v": torch.randn(b, skv, hk, d, device="cuda", dtype=dtype),
        "do": torch.randn(b, sq, hq, d, device="cuda", dtype=dtype),
    }
    if thd:
        for name, tensor in values.items():
            lengths = qlens if name in ("q", "do") else klens
            values[name] = torch.cat([tensor[i, :n] for i, n in enumerate(lengths)])
    expected = {name: [] for name in ("o", "stats", "dq", "dk", "dv")}
    qstart = kstart = 0
    with torch.enable_grad():
        for batch, (qlen, klen) in enumerate(zip(qlens, klens)):

            def part(name, start, length):
                value = values[name]
                return value[start : start + length] if thd else value[batch, :length]

            q = part("q", qstart, qlen).double().requires_grad_(backward)
            k = part("k", kstart, klen).double().requires_grad_(backward)
            v = part("v", kstart, klen).double().requires_grad_(backward)
            scores = q.transpose(0, 1) @ k.repeat_interleave(hq // hk, dim=1).permute(1, 2, 0) / math.sqrt(d)
            out = (scores.softmax(-1) @ v.repeat_interleave(hq // hk, dim=1).transpose(0, 1)).transpose(0, 1)
            expected["o"].append(out.detach())
            expected["stats"].append(scores.logsumexp(-1).transpose(0, 1).detach())
            if backward:
                gradients = torch.autograd.grad(out, (q, k, v), part("do", qstart, qlen).double())
                for name, gradient in zip(("dq", "dk", "dv"), gradients):
                    expected[name].append(gradient)
            qstart += qlen
            kstart += klen
    expected = {name: torch.cat(parts) for name, parts in expected.items() if parts}
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    bindings, outputs = {}, {}

    def ragged(node, lengths, width):
        offsets = torch.tensor((0, lengths[0] * width, sum(lengths) * width), device="cuda", dtype=torch.int64 if offset64 else torch.int32)
        offsets = offsets.reshape(b + 1, 1, 1, 1)
        desc = graph.tensor_like(offsets)
        node.set_ragged_offset(desc)
        bindings[desc] = offsets

    def descriptor(name, tensor, heads, length, lengths, node=None):
        is_stats = name == "stats"
        width = 1 if is_stats else d
        strides = (length * heads * width, width, heads * width, 1)
        if node is None:
            node = graph.tensor(dim=(b, heads, length, width), stride=strides, data_type=cudnn.data_type.FLOAT if is_stats else io, name=name)
        else:
            node.set_output(True).set_dim((b, heads, length, width)).set_stride(strides).set_data_type(cudnn.data_type.FLOAT if is_stats else io)
            outputs[name] = tensor
        if thd:
            ragged(node, lengths, heads * width)
        bindings[node] = tensor
        return node

    q = descriptor("q", values["q"], hq, sq, qlens)
    k = descriptor("k", values["k"], hk, skv, klens)
    v = descriptor("v", values["v"], hk, skv, klens)
    lengths = []
    for lens in (qlens, klens):
        storage = torch.tensor(lens, device="cuda", dtype=torch.int32).reshape(b, 1, 1, 1)
        node = graph.tensor_like(storage)
        bindings[node] = storage
        lengths.append(node)
    kwargs = dict(q=q, k=k, v=v, attn_scale=1 / math.sqrt(d), use_padding_mask=True, seq_len_q=lengths[0], seq_len_kv=lengths[1])
    if thd:
        kwargs.update(max_total_seq_len_q=sum(qlens), max_total_seq_len_kv=sum(klens))
    if backward:
        assert thd and stats
        o = descriptor("o", expected["o"].to(dtype), hq, sq, qlens)
        lse = descriptor("stats", expected["stats"].float(), hq, sq, qlens)
        do = descriptor("do", values["do"], hq, sq, qlens)
        nodes = graph.sdpa_backward(o=o, stats=lse, dO=do, **kwargs)
        for name, node, source, heads, length, lens in zip(("dq", "dk", "dv"), nodes, ("q", "k", "v"), (hq, hk, hk), (sq, skv, skv), (qlens, klens, klens)):
            descriptor(name, torch.empty_like(values[source]), heads, length, lens, node)
    else:
        o, lse = graph.sdpa(generate_stats=stats, **kwargs)
        descriptor("o", torch.empty_like(values["q"]), hq, sq, qlens, o)
        if stats:
            shape = (sum(qlens), hq) if thd else (b, sq, hq)
            descriptor("stats", torch.empty(shape, device="cuda", dtype=torch.float32), hq, sq, qlens, lse)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    native, excluded = [], []
    for i in range(graph.get_execution_plan_count()):
        if is_backend_engine(graph.get_engine_and_knobs_at_index(i)[0]):
            native.append(i)
        else:
            excluded.append(graph.get_plan_name_at_index(i))
    graph.deselect_engines(excluded)
    assert native, "the representative graph must retain native A-mode candidates"

    def check():
        torch.cuda.current_stream().synchronize()
        for name, output in outputs.items():
            lens = klens if name in ("dk", "dv") else qlens
            actual = output if thd else torch.cat([output[i, :n] for i, n in enumerate(lens)])
            atol = 0.002 if name == "stats" or dtype == torch.float16 else 0.006
            torch.testing.assert_close(actual.double(), expected[name], atol=atol, rtol=0 if name == "stats" else 0.02, msg=lambda msg: f"{name}: {msg}")

    return graph, bindings, outputs, native, check


def _exercise(cudnn_handle, dtype, sq=1, skv=300, stats=True, thd=True, offset64=False, backward=False):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("native candidate regression qualified on SM100 and SM103")
    stream = torch.cuda.ExternalStream(cudnn.get_stream(cudnn_handle))
    with torch.cuda.stream(stream):

        def case():
            return _case(cudnn_handle, dtype, sq, skv, stats, thd, offset64, backward)

        graph, bindings, outputs, native, check = case()
        compilation_failures, built = [], 0
        for index in native:
            try:
                graph.build_plan_at_index(index)
            except cudnn.cudnnGraphNotSupportedError as exc:
                if "CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED" in str(exc):
                    compilation_failures.append(f"{graph.get_plan_name_at_index(index)}: {exc}")
                continue
            workspace = torch.empty(max(1, graph.get_workspace_size_plan_at_index(index)), device="cuda", dtype=torch.uint8)
            for output in outputs.values():
                output.fill_(float("nan"))
            graph.execute_plan_at_index(bindings, workspace, index=index, handle=cudnn_handle)
            check()
            built += 1
        assert built > 0, "at least one retained native candidate must execute correctly"
        for policy in (None, cudnn.build_plan_policy.ALL):
            graph, bindings, outputs, native, check = case()
            graph.build_plans(*(() if policy is None else (policy,)))
            assert graph._plan_index in native
            workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
            for output in outputs.values():
                output.fill_(float("nan"))
            graph.execute(bindings, workspace, handle=cudnn_handle)
            check()
        # Check legal alternatives before reporting a bad candidate. A backend
        # fallback must not make a broken first candidate invisible to this test.
        assert not compilation_failures, "native candidates failed compilation (#1084):\n" + "\n".join(compilation_failures)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "sq,skv,stats,thd,offset64",
    [
        (1, 128, True, True, False),
        (1, 300, True, True, False),
        (1, 300, False, True, False),
        (1, 300, True, True, True),
        (1, 300, True, False, False),
        (2, 300, True, True, False),
    ],
    ids=["aligned", "tail", "no_stats", "offset64", "dense_control", "prefill_control"],
)
def test_native_decode_candidates(cudnn_handle, dtype, sq, skv, stats, thd, offset64):
    _exercise(cudnn_handle, dtype, sq, skv, stats, thd, offset64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("skv", [128, 300])
def test_native_decode_backward_candidates(cudnn_handle, dtype, skv):
    _exercise(cudnn_handle, dtype, skv=skv, backward=True)
