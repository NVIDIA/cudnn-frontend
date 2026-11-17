# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restored plans retain their public identity without heuristic candidates."""

import cudnn
import pytest
import torch

from cudnn.engines import is_backend_engine

pytestmark = pytest.mark.L0


def _execute(graph, bindings, workspace, index, api, handle):
    if api == "raw":
        pointers = {(node if isinstance(node, int) else node.get_uid()): value.data_ptr() for node, value in bindings.items()}
        graph._execute_plan_at_index(pointers, workspace.data_ptr(), index, handle, workspace_bytes=workspace.numel())
    else:
        graph.execute_plan_at_index(bindings, workspace, index=index, handle=handle)


def _graph_case(handle, api, kind, dtype):
    cls = cudnn.pygraph if api == "python" else cudnn._pybind_module.backend_graph
    native_handle = handle if api == "python" else handle.backend_handle
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cls(handle=native_handle, io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bindings, outputs, references = {}, {}, {}

    def tensor(value):
        node = graph.tensor_like(value)
        node.set_uid(len(bindings) + 1)
        bindings[node] = value
        return node

    def output(node, value, reference):
        node.set_uid(len(bindings) + 1)
        node.set_output(True).set_dim(value.shape).set_stride(value.stride())
        node.set_data_type(cudnn.data_type.FLOAT if value.dtype == torch.float32 else io)
        bindings[node] = value
        outputs[node.get_uid()] = value
        references[node.get_uid()] = reference

    torch.manual_seed(1742)
    if kind == "matmul":
        a = torch.randn(2, 16, 64, device="cuda", dtype=dtype)
        b = torch.randn(2, 64, 32, device="cuda", dtype=dtype)
        out = graph.matmul(A=tensor(a), B=tensor(b))
        output(out, torch.empty(2, 16, 32, device="cuda", dtype=dtype), a.double() @ b.double())
    else:
        q = torch.randn(2, 2, 17, 64, device="cuda", dtype=dtype) * 0.25
        k = torch.randn(2, 2, 65, 64, device="cuda", dtype=dtype) * 0.25
        v = torch.randn_like(k)
        backward = kind == "backward"
        with torch.enable_grad():
            qr, kr, vr = [x.double().requires_grad_(backward) for x in (q, k, v)]
            scores = qr @ kr.transpose(-2, -1) * 0.125
            reference = scores.softmax(-1) @ vr
            stats = scores.logsumexp(-1, keepdim=True)
            kwargs = dict(q=tensor(q), k=tensor(k), v=tensor(v), attn_scale=0.125)
            if backward:
                do = torch.randn_like(q)
                expected = torch.autograd.grad(reference, (qr, kr, vr), do.double())
                nodes = graph.sdpa_backward(
                    o=tensor(reference.detach().to(dtype)), stats=tensor(stats.detach().float()), dO=tensor(do), use_deterministic_algorithm=True, **kwargs
                )
                for node, value, ref in zip(nodes, (q, k, v), expected):
                    output(node, torch.empty_like(value), ref)
            else:
                out, lse = graph.sdpa(generate_stats=True, **kwargs)
                output(out, torch.empty_like(q), reference.detach())
                output(lse, torch.empty_like(stats, dtype=torch.float32), stats.detach())
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    return graph, cls, native_handle, bindings, outputs, references


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("as_bytes", [False, True], ids=["byte_list", "bytes"])
@pytest.mark.parametrize(
    "api,kind,provider",
    [(api, kind, "backend") for api in ("raw", "python") for kind in ("matmul", "forward", "backward")]
    + [("python", "forward", "frost"), ("raw", "matmul", "backend_defaults"), ("python", "matmul", "backend_defaults")],
    ids=[
        "raw_matmul",
        "raw_forward",
        "raw_backward",
        "python_matmul",
        "python_forward",
        "python_backward",
        "frost_forward",
        "raw_matmul_defaults",
        "python_matmul_defaults",
    ],
)
def test_restored_plan_metadata(cudnn_handle, monkeypatch, dtype, as_bytes, api, kind, provider):
    cc = torch.cuda.get_device_capability()
    if cudnn.backend_version() < 92400 or cc[0] < 8:
        pytest.skip("representative graphs require cuDNN 9.24+ and Ampere+")
    if provider == "backend_defaults" and cudnn.backend_version() < 93000:
        pytest.skip("compiled FULL_PRECISION_MATH defaults require cuDNN 9.30+")
    if provider == "frost":
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

        installed, version = cutedsl_state()
        if not installed or cutedsl_too_old(version) or (cc[0] == 8 and cc != (8, 0)):
            pytest.skip("requires a supported FROST target and CuTe DSL")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1" if provider == "frost" else "0")
    stream = torch.cuda.ExternalStream(cudnn.get_stream(cudnn_handle))
    with torch.cuda.stream(stream):
        graph, cls, native_handle, bindings, outputs, refs = _graph_case(cudnn_handle, api, kind, dtype)
        for index in range(graph.get_execution_plan_count()):
            record = graph.get_engine_and_knobs_at_index(index)
            if is_backend_engine(record[0]) != (provider != "frost"):
                continue
            if provider == "backend_defaults":
                domains = graph.get_knobs_for_engine(record[0])
                assert all(knob.type != cudnn.knob_type.NOT_SET for knob in domains)
                if not any(int(knob.type) == 33 for knob in domains):
                    continue
            try:
                graph.build_plan_at_index(index)
            except (cudnn.cudnnGraphNotSupportedError, NotImplementedError):
                continue
            break
        else:
            pytest.fail(f"no {provider} plan built for the representative {kind} graph")
        expected_name = graph.get_plan_name_at_index(index)
        expected_record = graph.get_engine_and_knobs_at_index(index)
        expected_notes = graph.get_behavior_notes_for_plan_at_index(index)
        workspace = torch.empty(max(1, graph.get_workspace_size_plan_at_index(index)), device="cuda", dtype=torch.uint8)
        _execute(graph, bindings, workspace, index, api, native_handle)
        stream.synchronize()
        for uid, value in outputs.items():
            torch.testing.assert_close(value.double(), refs[uid], atol=0.025 if dtype == torch.bfloat16 else 0.004, rtol=0.02)
        original = {uid: value.clone() for uid, value in outputs.items()}
        blob = graph.serialize(as_bytes=as_bytes)
        restored = cls(handle=native_handle)
        restored.deserialize(native_handle, blob, enforce_precompiled=True)
        assert restored.get_execution_plan_count() == 1
        errors = []
        queried = {}
        for label, query, expected in (
            ("name", restored.get_plan_name_at_index, expected_name),
            ("engine_and_knobs", restored.get_engine_and_knobs_at_index, expected_record),
        ):
            try:
                actual = query(0)
            except (RuntimeError, cudnn.cudnnGraphNotSupportedError) as exc:
                errors.append(f"{label}: {exc}")
            else:
                queried[label] = actual
            for bad in (-1, 1):
                with pytest.raises((ValueError, IndexError, RuntimeError, cudnn.cudnnGraphNotSupportedError), match="index|Index"):
                    query(bad)
        if not errors:
            actual_engine, actual_knobs = queried["engine_and_knobs"]
            assert actual_engine == expected_record[0]
            assert cudnn.knob_type.NOT_SET not in actual_knobs
            if provider == "backend_defaults":
                assert any(int(knob) == 33 for knob in actual_knobs)
            if (actual_engine, actual_knobs) == expected_record:
                assert queried["name"] == expected_name
            assert all(actual_knobs.get(key) == value for key, value in expected_record[1].items())
            # A backend can materialize default knobs while compiling/restoring.
            # The complete reported identity must rebuild the same executable.
            replay, _, _, replay_bindings, replay_outputs, _ = _graph_case(cudnn_handle, api, kind, dtype)
            replay.create_execution_plan(actual_engine, actual_knobs)
            replay_index = replay.get_execution_plan_count() - 1
            assert replay.get_plan_name_at_index(replay_index) == queried["name"]
            assert replay.get_engine_and_knobs_at_index(replay_index) == (actual_engine, actual_knobs)
            replay.build_plan_at_index(replay_index)
            assert replay.get_workspace_size_plan_at_index(replay_index) == restored.get_workspace_size_plan_at_index(0)
            replay_workspace = torch.empty(max(1, replay.get_workspace_size_plan_at_index(replay_index)), device="cuda", dtype=torch.uint8)
            for value in replay_outputs.values():
                value.fill_(float("nan"))
            _execute(replay, replay_bindings, replay_workspace, replay_index, api, native_handle)
            stream.synchronize()
            for uid, value in replay_outputs.items():
                assert torch.equal(value.contiguous().view(torch.uint8), original[uid].contiguous().view(torch.uint8))
        assert restored.get_behavior_notes_for_plan_at_index(0) == expected_notes
        assert restored.get_workspace_size_plan_at_index(0) == graph.get_workspace_size_plan_at_index(index)
        rebound = {node.get_uid(): value.clone() for node, value in bindings.items()}
        for uid in outputs:
            rebound[uid].fill_(float("nan"))
        _execute(restored, rebound, workspace, 0, api, native_handle)
        stream.synchronize()
        for uid, expected in original.items():
            assert torch.equal(rebound[uid].contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
        # Execute with fresh pointers before reporting query errors, so a metadata
        # regression cannot obscure the legal restored-execution control.
        assert not errors, "\n".join(errors)
