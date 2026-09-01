"""The Python half of the CPU-cost table.

Same kernel as bench_cpu_costs.cpp -- the CuTeDSL TMA tile add out of the same
container -- so the C++ and Python rows are directly comparable and the table
measures call paths rather than kernels.

    python bench_cpu_costs.py [mykernels.cudnn]

Method matches the C++ half: submit time with no synchronization inside a burst,
arms round-robin within each burst so drift hits all of them equally, the queue
drained BETWEEN bursts outside the timed region, and the median reported. A
repeated arm gives the noise floor.
"""

import json
import statistics
import sys
import time

import torch

import cudnn

BURSTS = 41
CALLS = 100
WARMUP = 300


def burst(fn):
    t0 = time.perf_counter()
    for _ in range(CALLS):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()  # drain between bursts, outside the timing
    return (t1 - t0) * 1e6 / CALLS


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    path = sys.argv[1] if len(sys.argv) > 1 else "mykernels.cudnn"

    with open(path + ".bench.json") as f:
        bench = json.load(f)
    kernel, symbol = bench["kernel"], bench["symbol"]
    shape = tuple(bench["shape"])
    # The TMA kernel takes the stream positionally; the plain add reads the
    # tvm-ffi environment stream and takes three arguments.
    stream_arg = bench.get("stream_arg", False)

    handle = cudnn.create_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)

    a = torch.randn(*shape, device="cuda", dtype=torch.float32)
    b = torch.randn(*shape, device="cuda", dtype=torch.float32)
    c = torch.zeros(*shape, device="cuda", dtype=torch.float32)

    arms = []

    # --- flow 2: the frontend front door, from a container -------------------
    lib = cudnn.import_from_disk(path, handle=handle)
    graph = lib[kernel]
    ws = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    ptrs = [a.data_ptr(), b.data_ptr(), c.data_ptr()]
    assert len(graph.variant_pack_uids_sorted()) == 3
    arms.append(("fe_python     (FE graph.execute, from container)", lambda: graph.execute(ptrs, ws, handle=handle)))

    # --- the same kernel with no frontend in the call ------------------------
    # Imported here rather than at module scope: this is the toolchain the
    # deploy box does not have, and only the benchmark needs it.
    import cuda.bindings.driver as cuda
    import tvm_ffi

    module = tvm_ffi.load_module(path + ".bench.so")
    fn = module[symbol]
    raw_stream = torch.cuda.current_stream().cuda_stream
    if stream_arg:
        arms.append(("ffi_python    (cutlass AOT .so, torch tensors)", lambda: fn(a, b, c, raw_stream)))
    else:
        arms.append(("ffi_python    (cutlass AOT .so, torch tensors)", lambda: fn(a, b, c)))

    # --- flow 1: the imperative path, JIT-compiled in this process -----------
    # Same device code; the difference is entirely in how the call gets there.
    import cutlass
    import cutlass.cute as cute

    from cudnn.engines.cutedsl_tma_add_engine import build_kernel

    compiled = cute.compile(
        build_kernel(),
        *[cute.runtime.make_fake_compact_tensor(cutlass.Float32, shape, stride_order=(1, 0)) for _ in range(3)],
        cuda.CUstream(0),
        options="--enable-tvm-ffi",
    )
    cu_stream = cuda.CUstream(raw_stream)
    arms.append(("py_imperative (CuTeDSL JIT object, torch tensors)", lambda: compiled(a, b, c, cu_stream)))

    # Noise floor: a repeat of the first arm.
    arms.append(("control       (= fe_python, repeated)", lambda: graph.execute(ptrs, ws, handle=handle)))

    for _ in range(WARMUP):
        for _label, fn_ in arms:
            fn_()
    torch.cuda.synchronize()

    samples = {label: [] for label, _ in arms}
    for _ in range(BURSTS):
        for label, fn_ in arms:
            samples[label].append(burst(fn_))

    print(f"kernel {kernel}, shape={shape}, {BURSTS} bursts x {CALLS} calls, round-robin, medians\n")
    print(f"  {'call path':<46} {'us/call':>10}")
    for label, _ in arms:
        print(f"  {label:<46} {statistics.median(samples[label]):>10.3f}")

    fe = statistics.median(samples[arms[0][0]])
    ffi_ = statistics.median(samples[arms[1][0]])
    print(f"\n  front door from Python: fe_python - ffi_python = {fe - ffi_:+.3f} us")


if __name__ == "__main__":
    main()
