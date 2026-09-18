# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real paired MoE graph plans: two retained parents and live captured inputs."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import hashlib
import json

import pytest
import torch

pytestmark = [
    pytest.mark.L1,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="paired MoE requires SM100"),
]

cases = [
    dict(name="expert257", experts=257, rows=8, k=64, n=64, sizes=[1] * 7 + [0] * 249 + [1], pitched=True),
    # Exercise token-tile boundaries and multiple token tiles per expert.
    dict(name="nine_rows", experts=4, rows=9, k=128, n=64, sizes=[4, 0, 5, 0], pitched=False),
    dict(name="seventeen_rows", experts=4, rows=17, k=128, n=128, sizes=[0, 17, 0, 0], pitched=True),
    dict(name="target_r64", experts=128, rows=64, k=2048, n=768, sizes=[1] * 64 + [0] * 64, pitched=False),
    dict(name="target_r512", experts=128, rows=512, k=2048, n=768, sizes=[4] * 128, pitched=False),
    dict(name="skew_r513", experts=128, rows=513, k=128, n=64, sizes=[8, 9, 17, 64, 415] + [0] * 123, pitched=True),
    dict(name="compact", experts=4, rows=8, k=128, n=64, sizes=[2, 0, 3, 3], pitched=False),
    dict(name="pitched_square", experts=5, rows=7, k=128, n=128, sizes=[0, 1, 0, 3, 3], pitched=True),
    dict(name="target_sparse", experts=128, rows=8, k=2048, n=768, sizes=[1] * 7 + [0] * 120 + [1], pitched=False),
    dict(name="one_token", experts=3, rows=1, k=64, n=64, sizes=[0, 1, 0], pitched=False),
    dict(name="one_active_expert", experts=2, rows=8, k=2048, n=768, sizes=[8, 0], pitched=False),
    dict(name="five_tokens", experts=7, rows=5, k=256, n=192, sizes=[0, 0, 2, 0, 0, 3, 0], pitched=False),
    # Reuse accumulator and scheduler slots across multiple persistent waves.
    dict(name="multi_wave", experts=128, rows=8, k=768, n=2048, sizes=[1] * 7 + [0] * 120 + [1], pitched=False),
    dict(name="many_waves", experts=128, rows=8, k=128, n=8192, sizes=[1] * 7 + [0] * 120 + [1], pitched=False),
]


@pytest.mark.parametrize("spec", cases, ids=[case["name"] for case in cases])
def test_paired_moe_native_graph_and_live_captures(spec, tmp_path):
    import cudnn
    from cudnn.frost import buffers

    _, version = buffers.cutedsl_state()
    if buffers.cutedsl_too_old(version):
        pytest.skip(buffers.cutedsl_requirement_error("paired MoE test"))
    root = Path(cudnn.__file__).resolve().parents[2]
    raw = tmp_path
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = dict(cases=[], checks=0, negatives=0)
    from cutlass import cute
    from cudnn.gemm.frost.moe_pair import pair_knobs
    from cudnn.frost import template_loader
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
    from cuda.bindings import driver as drv

    assert torch.cuda.get_device_capability() == (10, 0)
    generator = torch.Generator().manual_seed(1810)
    bf16, fp32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
    knobs = pair_knobs()

    def forbidden_empty(*args, **kwargs):
        raise AssertionError("execute allocated torch.empty")

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("execute compiled a kernel")

    @contextmanager
    def allocation_and_compile_guards():
        # Build the guards before capture. MagicMock construction inside
        # capture triggered cyclic GC and invalidated a retained graph test.
        with patch.object(torch, "empty", new=forbidden_empty), patch.object(cute, "compile", new=forbidden_compile):
            yield

    @contextmanager
    def execute_contract():
        old = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(old)

    def checked(call):
        error, *values = call
        assert int(error) == 0, str(error)
        return values

    # CUDA graph inspection follows the NVIDIA FlashInfer contract probe.
    def kernel_names(graph):
        handle = drv.CUgraph(graph.raw_cuda_graph())
        _, count = checked(drv.cuGraphGetNodes(handle, 0))
        nodes, actual_count = checked(drv.cuGraphGetNodes(handle, count))
        assert actual_count == count == 1
        names = []
        for node in nodes:
            (kind,) = checked(drv.cuGraphNodeGetType(node))
            assert kind == drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
            (params,) = checked(drv.cuGraphKernelNodeGetParams(node))
            (name,) = checked(drv.cuFuncGetName(params.func))
            names.append(name.decode())
        assert all("cudnn" in name and "frost_sm100_moe_swiglu_pair" in name and "sched_static" in name for name in names)
        assert all(("small_row_sched" in name) == (spec["rows"] <= 8) for name in names)
        return names

    def reference(x, w, offsets):
        out = torch.empty((x.shape[0], w.shape[1] // 2), dtype=torch.bfloat16)
        for expert, (start, end) in enumerate(zip(offsets, offsets[1:])):
            if start == end:
                continue
            gate, up = (x[start:end].float() @ w[expert].float().T).chunk(2, -1)
            out[start:end] = (up * (gate / (1.0 + torch.exp(-gate)))).to(torch.bfloat16)
        return out

    def close(actual, expected):
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        rel = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))
        assert rel <= 0.01, rel
        return rel

    def store(tensor, name):
        path = raw / name
        path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        return dict(path=str(path), sha256=sha(path), elements=tensor.numel())

    e, r, k, n = (spec[key] for key in ("experts", "rows", "k", "n"))
    starts = [0]
    for size in spec["sizes"]:
        starts.append(starts[-1] + size)
    assert len(starts) == e + 1 and starts[-1] == r
    pitch = k + (8 if spec["pitched"] else 0)
    expert_pitch = 2 * n * pitch + (8 if spec["pitched"] else 0)
    stride = [expert_pitch, 1, pitch]
    # Pitched case deliberately has 16B but not 32B expert alignment.
    assert not spec["pitched"] or expert_pitch * 2 % 32 == 16
    host_x = torch.randn((r, k), dtype=torch.bfloat16, generator=generator)
    host_w = [(torch.randn((e, 2 * n, k), dtype=torch.float32, generator=generator) / k**0.5).to(torch.bfloat16) for _ in range(2)]
    original_x = host_x.clone()
    original_weight = host_w[0][spec["sizes"].index(next(s for s in spec["sizes"] if s))].clone()
    first_expert = next(i for i, s in enumerate(spec["sizes"]) if s)
    stream = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=stream.cuda_stream)
    with torch.cuda.stream(stream):
        x = host_x.cuda()
        offsets = torch.tensor(starts[:-1], dtype=torch.int32, device="cuda").view(e, 1, 1)
        weights = []
        for hw in host_w:
            storage = torch.empty(e * expert_pitch, dtype=torch.bfloat16, device="cuda")
            storage.fill_(float("nan"))
            weight = storage.as_strided((e, 2 * n, k), (expert_pitch, pitch, 1))
            weight.copy_(hw)
            weights.append(weight)
        outputs = [torch.empty((r, n), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    stream.synchronize()
    g = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=fp32, compute_data_type=fp32, handle=handle)
    tx = g.tensor(name="tokens", dim=[1, r, k], stride=[r * k, k, 1], data_type=bf16)
    tw = g.tensor(name="parent_weight", dim=[e, k, 2 * n], stride=stride, data_type=bf16)
    to = g.tensor(name="offsets", dim=[e, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    gate = g.slice(tw, [slice(None), slice(None), slice(0, n)], name="gate_weight").set_stride(stride)
    up = g.slice(tw, [slice(None), slice(None), slice(n, 2 * n)], name="up_weight").set_stride(stride)
    mg = g.moe_grouped_matmul(tx, gate, to, mode=cudnn.moe_grouped_matmul_mode.NONE, name="gate")
    mu = g.moe_grouped_matmul(tx, up, to, mode=cudnn.moe_grouped_matmul_mode.NONE, name="up")
    ty = g.mul(g.swish(mg, name="silu"), mu, name="out").set_dim([1, r, n]).set_stride([r * n, n, 1]).set_output(True).set_data_type(bf16)
    chain, binding = analyze_with_binding(g)
    assert binding.bound_tensors() == [tx, tw, ty, to] and len(binding.operand_slices) == 2
    g.validate()
    g.build_operation_graph()
    g.create_execution_plan(20401, knobs.to_public())
    g.check_support()
    with patch.object(template_loader, "load_template", wraps=template_loader.load_template) as loads:
        g.build_plan_at_index(0)
    assert loads.call_count == 1
    template_path, template_params = loads.call_args.args[:2]
    assert Path(template_path).resolve() == root / "python/cudnn/gemm/frost/sm100/kernel_templates/sm100_moe_swiglu_pair.py"
    assert template_params.small_rows == (r <= 8)
    assert template_params.grid_ctas == torch.cuda.get_device_properties(0).multi_processor_count
    result.setdefault("templates", []).append(dict(path=template_path, sha256=sha(template_path), params=repr(template_params)))
    engine, actual_knobs = g.get_engine_and_knobs_at_index(0)
    assert int(engine) == 20401 and actual_knobs == knobs.to_public()
    workspace_bytes = g.get_workspace_size_plan_at_index(0)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    packs = [{tx: x.unsqueeze(0), tw: w.transpose(1, 2), to: offsets, ty: y.unsqueeze(0)} for w, y in zip(weights, outputs)]
    graphs = []
    case = dict(
        **spec,
        parent_stride=stride,
        engine=int(engine),
        small_rows=template_params.small_rows,
        workspace_bytes=workspace_bytes,
        parent_ptrs=[w.data_ptr() for w in weights],
        checks=[],
        negative_controls=[],
        kernels=[],
    )
    result["cases"].append(case)
    assert weights[0].data_ptr() != weights[1].data_ptr()

    def launch(index):
        with execute_contract():
            g.execute_plan_at_index(packs[index], workspace, 0, handle=handle)

    def observe(index, label, bounds=starts):
        stream.synchronize()
        actual = outputs[index].cpu()
        expected = reference(host_x, host_w[index], bounds)
        rel = close(actual, expected)
        prefix = f'{spec["name"]}_{len(case["checks"])}'
        case["checks"].append(
            dict(label=label, pack=index, rel_l2=rel, actual=store(actual, prefix + "_actual.bin"), expected=store(expected, prefix + "_expected.bin"))
        )
        result["checks"] += 1
        return expected

    def replay(index, label, bounds=starts):
        with torch.cuda.stream(stream):
            outputs[index].fill_(float("nan"))
            workspace.fill_(0xA5)
            graphs[index].replay()
        return observe(index, label, bounds)

    expected_initial = []
    for index in range(2):
        with allocation_and_compile_guards(), torch.cuda.stream(stream):
            outputs[index].fill_(float("nan"))
            workspace.fill_(0xA5)
            launch(index)
        expected_initial.append(observe(index, "eager"))
        captured = torch.cuda.CUDAGraph(keep_graph=True)
        with allocation_and_compile_guards(), torch.cuda.graph(captured, stream=stream):
            launch(index)
        graphs.append(captured)
        case["kernels"].append(kernel_names(captured))
        replay(index, "capture_replay")
    replay(0, "retained_after_other_pack")
    for label in ("tokens", "weights", "offsets"):
        bounds = starts
        with torch.cuda.stream(stream):
            if label == "tokens":
                host_x.neg_()
                x.copy_(host_x)
            elif label == "weights":
                host_w[0][first_expert].neg_()
                weights[0][first_expert].copy_(host_w[0][first_expert])
            else:
                bounds = [0] * e + [r]
                offsets.copy_(torch.tensor(bounds[:-1], dtype=torch.int32).view(e, 1, 1))
        expected = replay(0, "live_" + label, bounds)
        try:
            close(expected, expected_initial[0])
        except AssertionError:
            case["negative_controls"].append(dict(mutation=label, reference_changed=True))
            result["negatives"] += 1
        else:
            raise AssertionError("mutation did not exercise stale-input detection: " + label)
        with torch.cuda.stream(stream):
            if label == "tokens":
                host_x.copy_(original_x)
                x.copy_(host_x)
            elif label == "weights":
                host_w[0][first_expert].copy_(original_weight)
                weights[0][first_expert].copy_(original_weight)
            else:
                offsets.copy_(torch.tensor(starts[:-1], dtype=torch.int32).view(e, 1, 1))
    replay(0, "restored")
    replay(1, "other_pack_unchanged")
    assert len(case["checks"]) == 10 and len(case["negative_controls"]) == 3
    stream.synchronize()
    for captured in graphs:
        captured.reset()
    cudnn.destroy_handle(handle)

    assert result["checks"] == 10 and result["negatives"] == 3

    (raw / "fc1_result.json").write_text(json.dumps(result, indent=2) + "\n")
