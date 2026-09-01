"""AOT kernel export/import.

Covers the two AOT flows end to end: a container written on a build box and
executed in a process that never compiles anything, and the same graphs
published into the process-global table with no file at all.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

import cudnn

pytestmark = pytest.mark.L0

cutlass = pytest.importorskip("cutlass", reason="AOT export needs nvidia-cutlass-dsl")
tvm_ffi = pytest.importorskip("tvm_ffi", reason="AOT export needs apache-tvm-ffi")


@pytest.fixture(autouse=True)
def enable_demo_engines(monkeypatch):
    """The CuTeDSL demo family is opt_in, and engines are not injectable."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")


def build_add_graph(name, shape=(4, 1024), dtype=torch.float32):
    """A single-node elementwise-add graph routed to the CuTeDSL engine."""
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a = graph.tensor(dim=list(shape), stride=_contig_stride(shape), data_type=cudnn.data_type.FLOAT, name="A")
    b = graph.tensor(dim=list(shape), stride=_contig_stride(shape), data_type=cudnn.data_type.FLOAT, name="B")
    c = graph.add(a, b, name="sum")
    c.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    graph.check_support()
    graph.build_plans()
    graph.set_name(name)
    return graph, a, b, c


def _contig_stride(shape):
    stride, acc = [], 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return list(reversed(stride))


def test_cutedsl_engine_is_selected():
    """The vehicle must actually route to the CuTeDSL engine, not to cuDNN."""
    _requires_cuda()
    graph, *_ = build_add_graph("add_f32_selected")
    assert graph.selected_engine is not None
    assert graph.selected_engine.name == "cutedsl_pointwise_add"


def test_round_trip_matches_direct_execution():
    """serialize -> deserialize -> execute is numerically identical to executing
    the same graph directly."""
    _requires_cuda()
    shape = (4, 1024)
    handle = cudnn.create_handle()

    graph, a, b, c = build_add_graph("add_fwd_f32", shape)

    a_gpu = torch.randn(shape, device="cuda", dtype=torch.float32)
    b_gpu = torch.randn(shape, device="cuda", dtype=torch.float32)
    direct = torch.zeros(shape, device="cuda", dtype=torch.float32)

    graph.execute({a: a_gpu, b: b_gpu, c: direct}, None, handle=handle)
    torch.cuda.synchronize()

    data = cudnn.aot.serialize_graph(graph)
    assert len(data) > 0

    loaded = cudnn.aot.deserialize_graph(data, handle)
    assert loaded.get_name() == "add_fwd_f32"
    # The loaded graph must dispatch to the artifact, not to a cuDNN plan.
    assert loaded._lowered_graph._has_cutedsl_payload()

    uid_order = loaded.variant_pack_uids_sorted()
    assert sorted(uid_order) == sorted([a.uid, b.uid, c.uid])

    by_uid = {a.uid: a_gpu, b.uid: b_gpu}
    round_tripped = torch.zeros(shape, device="cuda", dtype=torch.float32)
    by_uid[c.uid] = round_tripped

    ws_size = loaded.get_workspace_size()
    ws = torch.empty(max(ws_size, 1), dtype=torch.uint8, device="cuda")

    # The doc's spelling: execute() takes the gathered pointer array directly.
    loaded.execute([by_uid[u] for u in uid_order], ws, handle=handle)
    torch.cuda.synchronize()

    torch.testing.assert_close(round_tripped, a_gpu + b_gpu, rtol=0, atol=0)
    torch.testing.assert_close(round_tripped, direct, rtol=0, atol=0)

    cudnn.destroy_handle(handle)


def test_concurrent_execute_of_one_imported_graph():
    """execute() does not mutate the graph, so one graph may be executed from
    any number of threads."""
    _requires_cuda()
    import threading

    shape = (4, 1024)
    handle = cudnn.create_handle()
    graph, a, b, c = build_add_graph("add_threads_f32", shape)
    loaded = cudnn.aot.deserialize_graph(cudnn.aot.serialize_graph(graph), handle)
    uid_order = loaded.variant_pack_uids_sorted()

    a_gpu = torch.randn(shape, device="cuda")
    b_gpu = torch.randn(shape, device="cuda")
    expected = a_gpu + b_gpu
    failures = []

    def run(i):
        out = torch.zeros(shape, device="cuda")
        by_uid = {a.uid: a_gpu, b.uid: b_gpu, c.uid: out}
        ws = torch.empty(1, dtype=torch.uint8, device="cuda")
        try:
            loaded.execute_ptrs([by_uid[u] for u in uid_order], ws, handle=handle)
            torch.cuda.synchronize()
            if not torch.equal(out, expected):
                failures.append(f"thread {i}: numerical mismatch")
        except Exception as e:  # noqa: BLE001 — reported, not swallowed
            failures.append(f"thread {i}: {e}")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures, failures

    cudnn.destroy_handle(handle)


def test_wrong_architecture_is_rejected_at_load():
    """A misread artifact must be a clean error, never an illegal access."""
    _requires_cuda()
    handle = cudnn.create_handle()
    graph, *_ = build_add_graph("add_arch_f32")
    data = cudnn.aot.serialize_graph(graph)

    tag = data.find(b"sm_100a")
    assert tag > 0, "expected the artifact to record its target architecture"
    tampered = bytearray(data)
    tampered[tag : tag + 7] = b"sm_90aa"

    with pytest.raises(Exception, match="but this device is"):
        cudnn.aot.deserialize_graph(bytes(tampered), handle)

    cudnn.destroy_handle(handle)


def test_deserialized_graph_refuses_to_be_rebuilt():
    """Build-time calls on a graph that came from an artifact must be a clear
    error, not undefined behaviour."""
    _requires_cuda()
    handle = cudnn.create_handle()
    graph, *_ = build_add_graph("add_guard_f32")
    loaded = cudnn.aot.deserialize_graph(cudnn.aot.serialize_graph(graph), handle)

    for call in (
        lambda: loaded._lowered_graph.validate(),
        lambda: loaded._lowered_graph.build_operation_graph(),
        lambda: loaded._lowered_graph.create_execution_plans([cudnn.heur_mode.A]),
        lambda: loaded._lowered_graph.check_support(),
        lambda: loaded._lowered_graph.build_plans(),
    ):
        with pytest.raises(Exception, match="deserialized"):
            call()

    cudnn.destroy_handle(handle)


def test_export_requires_a_name():
    _requires_cuda()
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    shape = (2, 256)
    a = graph.tensor(dim=list(shape), stride=_contig_stride(shape), data_type=cudnn.data_type.FLOAT, name="A")
    b = graph.tensor(dim=list(shape), stride=_contig_stride(shape), data_type=cudnn.data_type.FLOAT, name="B")
    c = graph.add(a, b, name="sum")
    c.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A])

    with pytest.raises(ValueError, match="set_name"):
        cudnn.aot.serialize_graph(graph)


def test_backend_engine_export_is_not_implemented(monkeypatch):
    """A cuDNN-backend graph must say so, not produce a broken artifact."""
    _requires_cuda()
    # Off, or the demo pointwise engine claims this graph and it never reaches
    # the backend.
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    shape = (2, 256)
    a = graph.tensor(dim=list(shape), stride=_contig_stride(shape), data_type=cudnn.data_type.FLOAT, name="A")
    b = graph.tensor(dim=list(shape), stride=_contig_stride(shape), data_type=cudnn.data_type.FLOAT, name="B")
    c = graph.add(a, b, name="sum")
    c.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A])
    graph.set_name("add_backend_f32")

    with pytest.raises(NotImplementedError, match="cuDNN backend"):
        cudnn.aot.serialize_graph(graph)


# ---------------------------------------------------------------------------
# Phase 2: a container of N graphs, executed in a process that never compiled
# anything.
# ---------------------------------------------------------------------------

# Runs in the fresh process: import the container, execute every kernel in it
# from the pointer array, and report the results as JSON on stdout.
_CONSUMER = r"""
import json, sys
import torch
import cudnn

path, spec_json = sys.argv[1], sys.argv[2]
spec = json.loads(spec_json)
handle = cudnn.create_handle()

lib = cudnn.import_from_disk(path, handle)
result = {"keys": sorted(lib.keys()), "outputs": {}}

for name, entry in spec.items():
    graph = lib[name]
    shape = tuple(entry["shape"])
    torch.manual_seed(entry["seed"])
    a = torch.randn(shape, device="cuda")
    b = torch.randn(shape, device="cuda")
    out = torch.full(shape, float("nan"), device="cuda")

    uid_order = graph.variant_pack_uids_sorted()
    by_uid = {entry["a_uid"]: a, entry["b_uid"]: b, entry["c_uid"]: out}
    ws = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute([by_uid[u] for u in uid_order], ws, handle=handle)
    torch.cuda.synchronize()

    result["outputs"][name] = {
        "exact": bool(torch.equal(out, a + b)),
        "uid_order": list(uid_order),
        "checksum": float(out.double().sum().item()),
    }

cudnn.destroy_handle(handle)
# The point of flow 2: no kernel toolchain ran in this process.
# nvidia_cutlass_dsl lands in sys.modules from the wheel's path hook alone;
# "cutlass" is the import that actually brings in the JIT.
result["compiler_loaded"] = any(m in sys.modules for m in ("cutlass", "cutlass.cute"))
print("RESULT " + json.dumps(result))
"""


def _run_consumer(path, spec):
    """Run the consumer in a genuinely fresh interpreter."""
    proc = subprocess.run(
        [sys.executable, "-c", _CONSUMER, path, json.dumps(spec)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert proc.returncode == 0, f"consumer failed:\n{proc.stdout}\n{proc.stderr}"
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line is not None, f"no result from consumer:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(line[len("RESULT ") :])


def test_container_of_n_graphs_runs_in_a_fresh_process(tmp_path):
    """The whole point of flow 2: build here, execute over there, no compiler."""
    _requires_cuda()
    path = str(tmp_path / "mykernels.cudnn")

    specs = {"add_small_f32": (2, 512), "add_medium_f32": (4, 1024), "add_large_f32": (8, 2048)}
    graphs, spec = [], {}
    for name, shape in specs.items():
        graph, a, b, c = build_add_graph(name, shape)
        graphs.append(graph)
        spec[name] = {
            "shape": list(shape),
            "seed": len(name),
            "a_uid": a.uid,
            "b_uid": b.uid,
            "c_uid": c.uid,
        }

    cudnn.export_to_disk(graphs, path=path)
    assert os.path.getsize(path) > 0

    result = _run_consumer(path, spec)
    assert result["keys"] == sorted(specs)
    assert not result["compiler_loaded"], "the consuming process should never load a kernel toolchain"

    for name in specs:
        got = result["outputs"][name]
        assert got["exact"], f"{name} did not match a+b in the fresh process"
        # And the same expectation reproduced here, from the same seed.
        torch.manual_seed(spec[name]["seed"])
        shape = tuple(spec[name]["shape"])
        a_gpu = torch.randn(shape, device="cuda")
        b_gpu = torch.randn(shape, device="cuda")
        assert got["checksum"] == pytest.approx((a_gpu + b_gpu).double().sum().item(), rel=0, abs=0)


def test_container_lookup_is_by_name(tmp_path):
    """Adding a kernel must not change what an existing caller resolves."""
    _requires_cuda()
    handle = cudnn.create_handle()

    small = str(tmp_path / "small.cudnn")
    big = str(tmp_path / "big.cudnn")

    g1, *_ = build_add_graph("kernel_one")
    cudnn.export_to_disk([g1], path=small)

    # A second container with an extra kernel sorted BEFORE the first one.
    g1b, *_ = build_add_graph("kernel_one")
    g0, *_ = build_add_graph("aaa_kernel_zero")
    cudnn.export_to_disk([g0, g1b], path=big)

    lib_small = cudnn.import_from_disk(small, handle)
    lib_big = cudnn.import_from_disk(big, handle)

    assert sorted(lib_small.keys()) == ["kernel_one"]
    assert sorted(lib_big.keys()) == ["aaa_kernel_zero", "kernel_one"]
    assert lib_big["kernel_one"].get_name() == "kernel_one"

    with pytest.raises(KeyError, match="no kernel named"):
        lib_big["nope"]

    cudnn.destroy_handle(handle)


def test_duplicate_names_in_one_call_are_rejected(tmp_path):
    _requires_cuda()
    g1, *_ = build_add_graph("same_name")
    g2, *_ = build_add_graph("same_name")
    with pytest.raises(ValueError, match="both named"):
        cudnn.export_to_disk([g1, g2], path=str(tmp_path / "dup.cudnn"))


def test_unnamed_graph_is_rejected(tmp_path):
    _requires_cuda()
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    shape = (2, 256)
    a = graph.tensor(dim=list(shape), stride=_contig_stride(shape), data_type=cudnn.data_type.FLOAT, name="A")
    b = graph.tensor(dim=list(shape), stride=_contig_stride(shape), data_type=cudnn.data_type.FLOAT, name="B")
    c = graph.add(a, b, name="sum")
    c.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A])
    with pytest.raises(ValueError, match="set_name"):
        cudnn.export_to_disk([graph], path=str(tmp_path / "unnamed.cudnn"))


def test_not_a_container_is_a_clean_error(tmp_path):
    _requires_cuda()
    junk = tmp_path / "junk.cudnn"
    junk.write_bytes(b"this is not a cudnn kernel library")
    with pytest.raises(Exception, match="cuDNN kernel library"):
        cudnn.import_from_disk(str(junk))


# ---------------------------------------------------------------------------
# Phase 3: register to memory. No file, same process, same execute API.
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_registry():
    """Leave the process table as we found it, whatever the test does."""
    before = set(cudnn.aot.registered_global_names())
    yield
    for name in set(cudnn.aot.registered_global_names()) - before:
        cudnn.aot.unregister_global(name)


def test_register_global_then_execute(clean_registry):
    """Flow 3 end to end: publish under a name, fetch it back, execute."""
    _requires_cuda()
    shape = (4, 1024)
    handle = cudnn.create_handle()

    graph, a, b, c = build_add_graph("ln_resident_f32", shape)
    cudnn.register_global(graph)
    assert "ln_resident_f32" in cudnn.aot.registered_global_names()

    fetched = cudnn.get_global("ln_resident_f32", handle)
    assert fetched.get_name() == "ln_resident_f32"

    a_gpu = torch.randn(shape, device="cuda")
    b_gpu = torch.randn(shape, device="cuda")
    out = torch.full(shape, float("nan"), device="cuda")

    uid_order = fetched.variant_pack_uids_sorted()
    by_uid = {a.uid: a_gpu, b.uid: b_gpu, c.uid: out}
    ws = torch.empty(max(fetched.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")

    # Identical to flow 2 from the lookup down.
    fetched.execute([by_uid[u] for u in uid_order], ws, handle=handle)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, a_gpu + b_gpu, rtol=0, atol=0)

    cudnn.unregister_global("ln_resident_f32")
    assert "ln_resident_f32" not in cudnn.aot.registered_global_names()

    cudnn.destroy_handle(handle)


def test_register_global_writes_no_file(tmp_path, clean_registry):
    """Flow 3's whole point: nothing is serialised."""
    _requires_cuda()
    cache = tmp_path / "aot_cache"
    old = os.environ.get("CUDNN_FRONTEND_AOT_CACHE_DIR")
    os.environ["CUDNN_FRONTEND_AOT_CACHE_DIR"] = str(cache)
    try:
        graph, *_ = build_add_graph("ln_nofile_f32")
        cudnn.register_global(graph)
        cudnn.get_global("ln_nofile_f32")
        assert not cache.exists(), "register_global() materialised a module file"
    finally:
        if old is None:
            os.environ.pop("CUDNN_FRONTEND_AOT_CACHE_DIR", None)
        else:
            os.environ["CUDNN_FRONTEND_AOT_CACHE_DIR"] = old


def test_duplicate_registration_needs_override(clean_registry):
    _requires_cuda()
    g1, *_ = build_add_graph("ln_dup_f32")
    cudnn.register_global(g1)

    g2, *_ = build_add_graph("ln_dup_f32", (8, 256))
    with pytest.raises(Exception, match="already registered"):
        cudnn.register_global(g2)

    cudnn.register_global(g2, override=True)


def test_override_and_refetch(clean_registry):
    """A handle from get_global is a snapshot: it keeps running the kernel it was
    fetched with, and a caller that wants the new one re-fetches."""
    _requires_cuda()
    handle = cudnn.create_handle()

    first_shape, second_shape = (4, 1024), (2, 64)
    g1, a1, b1, c1 = build_add_graph("ln_override_f32", first_shape)
    cudnn.register_global(g1)
    stale = cudnn.get_global("ln_override_f32", handle)
    stale_order = stale.variant_pack_uids_sorted()

    g2, a2, b2, c2 = build_add_graph("ln_override_f32", second_shape)
    cudnn.register_global(g2, override=True)

    fresh = cudnn.get_global("ln_override_f32", handle)

    def run(graph, shape, uids, order):
        a_gpu = torch.randn(shape, device="cuda")
        b_gpu = torch.randn(shape, device="cuda")
        out = torch.full(shape, float("nan"), device="cuda")
        by_uid = dict(zip(uids, (a_gpu, b_gpu, out)))
        ws = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
        graph.execute([by_uid[u] for u in order], ws, handle=handle)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, a_gpu + b_gpu, rtol=0, atol=0)

    # The stale snapshot still runs the kernel it was fetched with...
    run(stale, first_shape, (a1.uid, b1.uid, c1.uid), stale_order)
    # ...and the re-fetched handle runs the new one.
    run(fresh, second_shape, (a2.uid, b2.uid, c2.uid), fresh.variant_pack_uids_sorted())

    cudnn.destroy_handle(handle)


def test_get_unknown_global_is_a_clean_error(clean_registry):
    _requires_cuda()
    with pytest.raises(Exception, match="Nothing is registered under"):
        cudnn.get_global("no_such_kernel")
    with pytest.raises(Exception, match="Nothing is registered under"):
        cudnn.unregister_global("no_such_kernel")


def test_registry_holds_a_reference(clean_registry):
    """The compiled object must survive the builder dropping every reference."""
    _requires_cuda()
    import gc

    handle = cudnn.create_handle()
    graph, a, b, c = build_add_graph("ln_lifetime_f32", (2, 512))
    uids = (a.uid, b.uid, c.uid)
    cudnn.register_global(graph)

    del graph, a, b, c
    gc.collect()

    fetched = cudnn.get_global("ln_lifetime_f32", handle)
    shape = (2, 512)
    a_gpu = torch.randn(shape, device="cuda")
    b_gpu = torch.randn(shape, device="cuda")
    out = torch.zeros(shape, device="cuda")
    by_uid = dict(zip(uids, (a_gpu, b_gpu, out)))
    ws = torch.empty(1, dtype=torch.uint8, device="cuda")
    fetched.execute([by_uid[u] for u in fetched.variant_pack_uids_sorted()], ws, handle=handle)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a_gpu + b_gpu, rtol=0, atol=0)

    cudnn.destroy_handle(handle)


def test_wire_prefix_never_appears_in_user_code(clean_registry):
    """FE namespaces its entries as cudnn.<name>; the user says <name>."""
    _requires_cuda()
    import tvm_ffi

    graph, *_ = build_add_graph("ln_prefix_f32")
    cudnn.register_global(graph)

    assert cudnn.aot.registered_global_names() == ["ln_prefix_f32"] or "ln_prefix_f32" in cudnn.aot.registered_global_names()
    # One entry per launch step, named <prefix>.<i>; this kernel is one launch.
    assert tvm_ffi.get_global_func("cudnn.ln_prefix_f32.0", allow_missing=True) is not None
    assert tvm_ffi.get_global_func("ln_prefix_f32.0", allow_missing=True) is None

    cudnn.unregister_global("ln_prefix_f32")
    assert tvm_ffi.get_global_func("cudnn.ln_prefix_f32.0", allow_missing=True) is None
