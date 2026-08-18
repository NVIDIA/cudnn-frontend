// What does each way of reaching one kernel cost per launch, on the CPU?
//
// Every arm below ends in the SAME compiled CuTeDSL elementwise add, out of the
// same container, except the two floor arms, which are a hand-written kernel
// launched with no FFI at all (bench_native_add.cu). So the numbers compare
// CALL PATHS, not kernels -- which is the property the earlier front-door
// measurement lacked: it compared an FE call over a cuDNN backend plan against
// a tvm-ffi call over a cutlass kernel, two different dispatch stacks over two
// different kernels, and the two could not honestly be differenced.
//
//   ffi_module     ffi::Function out of the AOT .so, called directly
//   ffi_global     the same function resolved from the tvm-ffi global table
//   fe_container   fe::Graph from import_from_disk() -> graph.execute()
//   fe_global      fe::Graph from get_global()       -> graph.execute()
//   native_launch  hand-written kernel, cudaLaunchKernel
//   raw_cubin      the same kernel as a cubin, cuModuleLoad + cuLaunchKernel
//   graph_replay   cudaGraph capture of ffi_module, replayed
//
// fe_* minus ffi_* is the frontend's front door over an OSS-engine plan, which
// is the number this whole design turns on and which was previously estimated
// rather than measured.
//
// Method: submit time, so no synchronization inside a burst. Arms run
// round-robin within each burst so drift hits all of them equally, and the
// reported figure is the median over bursts. The queue is drained BETWEEN
// bursts, outside the timed region: without that it fills, every arm becomes
// bound by launch backpressure, and the host-side differences vanish into it.
// A repeated arm ("control") is reported to show what the noise floor is.
//
//     ./bench_cpu_costs [mykernels.cudnn]

#include <cudnn_frontend/kernel_library.h>

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/module.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace fe  = cudnn_frontend;
namespace ffi = tvm::ffi;

extern "C" void native_add_launch(void const *a, void const *b, void *c, int n, cudaStream_t stream);

#define CHECK_FE(expr)                                                                        \
    do {                                                                                      \
        auto _status = (expr);                                                                \
        if (_status.is_bad()) {                                                               \
            std::fprintf(stderr, "FAILED: %s\n  %s\n", #expr, _status.get_message().c_str()); \
            return 1;                                                                         \
        }                                                                                     \
    } while (0)

#define CHECK_CUDA(expr)                                                                      \
    do {                                                                                      \
        cudaError_t _err = (expr);                                                            \
        if (_err != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA FAILED: %s\n  %s\n", #expr, cudaGetErrorString(_err)); \
            return 1;                                                                         \
        }                                                                                     \
    } while (0)

#define CHECK_CU(expr)                                                              \
    do {                                                                            \
        CUresult _err = (expr);                                                     \
        if (_err != CUDA_SUCCESS) {                                                 \
            char const *_msg = nullptr;                                             \
            cuGetErrorString(_err, &_msg);                                          \
            std::fprintf(stderr, "CU FAILED: %s\n  %s\n", #expr, _msg ? _msg : ""); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

namespace {

std::string
manifest_string(std::string const &text, std::string const &key) {
    auto const at = text.find("\"" + key + "\"");
    if (at == std::string::npos) return {};
    auto const open  = text.find('"', text.find(':', at));
    auto const close = text.find('"', open + 1);
    if (open == std::string::npos || close == std::string::npos) return {};
    return text.substr(open + 1, close - open - 1);
}

double
median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Arm {
    char const *label;
    std::function<void()> call;
    std::vector<double> samples;
};

}  // namespace

int
main(int argc, char **argv) {
    std::string const container = (argc > 1) ? argv[1] : "mykernels.cudnn";

    std::ifstream mf(container + ".bench.json");
    if (!mf.good()) {
        std::fprintf(stderr, "cannot read %s.bench.json -- run flow2_export.py --bench first\n", container.c_str());
        return 1;
    }
    std::string const bench((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    std::string const kernel = manifest_string(bench, "kernel");
    std::string const symbol = manifest_string(bench, "symbol");
    // Some kernels take the stream as a positional handle rather than reading
    // the tvm-ffi environment stream, so the direct-FFI arms have to match the
    // signature the artifact actually exports.
    bool const stream_arg = bench.find("\"stream_arg\": true") != std::string::npos;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "could not create a cuDNN handle\n");
        return 1;
    }
    cudaStream_t stream = nullptr;
    cudnnGetStream(handle, &stream);

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));

    // ---------------------------------------------------------------- data --
    // Shape comes from the artifact, not from this file: the plain add is a
    // flat 1-D view, the TMA tile add is rank 2, and the direct-FFI arms must
    // hand the kernel exactly what it was compiled for.
    std::vector<int64_t> shape;
    {
        auto const at   = bench.find("\"shape\"");
        auto const open = bench.find('[', at);
        char const *p   = bench.c_str() + open + 1;
        while (true) {
            char *end       = nullptr;
            long long const v = std::strtoll(p, &end, 10);
            if (end == p) break;
            shape.push_back(v);
            p = end;
            while (*p == ' ' || *p == ',' || *p == '\n') p++;
            if (*p == ']') break;
        }
    }
    if (shape.empty()) {
        std::fprintf(stderr, "no shape in %s.bench.json -- re-run flow2_export.py --bench\n", container.c_str());
        return 1;
    }
    int64_t n = 1;
    for (auto d : shape) n *= d;

    float *da, *db, *dc;
    CHECK_CUDA(cudaMalloc(&da, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dc, n * sizeof(float)));

    // -------------------------------------------------- FE: from container --
    fe::KernelLibrary lib;
    CHECK_FE(fe::import_from_disk(container, &lib, handle));
    fe::graph::Graph from_container;
    CHECK_FE(lib.get(kernel, &from_container));

    auto const uid_order   = from_container.get_variant_pack_uids_sorted();
    int64_t workspace_size = 0;
    CHECK_FE(from_container.get_workspace_size(workspace_size));
    void *workspace = nullptr;
    if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    std::vector<void *> ptrs(uid_order.size());
    for (size_t i = 0; i < uid_order.size(); i++) {
        ptrs[i] = (i == 0) ? static_cast<void *>(da) : (i == 1) ? static_cast<void *>(db) : static_cast<void *>(dc);
    }

    // ----------------------------------------------- FE: from the registry --
    // Same graph, published under its name and fetched back, so the only
    // difference from fe_container is where the Graph came from.
    auto shared = std::make_shared<fe::graph::Graph>(from_container);
    CHECK_FE(fe::register_global(shared, true));
    fe::graph::Graph from_registry;
    CHECK_FE(fe::get_global(kernel, &from_registry));

    // ------------------------------------------------- tvm-ffi, no FE at all --
    ffi::Module module = ffi::Module::LoadFromFile(container + ".bench.so");
    auto module_fn     = module->GetFunction(symbol);
    if (!module_fn.has_value()) {
        std::fprintf(stderr, "the benchmark module does not export %s\n", symbol.c_str());
        return 1;
    }
    ffi::Function fn_from_module = module_fn.value();

    // Publish it under a name and resolve it back, which is what a C++ caller
    // does in the register-to-memory flow when it skips FE entirely.
    ffi::Function::SetGlobal("bench_cpu_costs_add", fn_from_module, true);
    auto global_fn = ffi::Function::GetGlobal("bench_cpu_costs_add");
    if (!global_fn.has_value()) {
        std::fprintf(stderr, "could not resolve the function back out of the global table\n");
        return 1;
    }
    ffi::Function fn_from_global = global_fn.value();

    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
    DLDataType const f32{kDLFloat, 32, 1};
    DLDevice const gpu{kDLCUDA, device};
    int32_t const ndim = static_cast<int32_t>(shape.size());
    DLTensor const proto_a{da, gpu, ndim, f32, shape.data(), strides.data(), 0};
    DLTensor const proto_b{db, gpu, ndim, f32, shape.data(), strides.data(), 0};
    DLTensor const proto_c{dc, gpu, ndim, f32, shape.data(), strides.data(), 0};

    // The kernel takes the tvm-ffi environment stream, so a direct caller has
    // to put its stream there -- exactly as FE's dispatch target does. Setting
    // it is part of the arm, not excluded from it.
    auto call_ffi = [&](ffi::Function const &f) {
        DLTensor t[3] = {proto_a, proto_b, proto_c};
        ffi::AnyView args[4];
        args[0] = &t[0];
        args[1] = &t[1];
        args[2] = &t[2];
        int n_args = 3;
        if (stream_arg) {
            args[3] = static_cast<void *>(stream);
            n_args  = 4;
        }
        // Set regardless: a kernel taking the stream positionally ignores this,
        // and one reading the environment stream needs it. Keeping both makes
        // the two arms comparable.
        void *original = nullptr;
        TVMFFIEnvSetStream(kDLCUDA, device, static_cast<void *>(stream), &original);
        ffi::Any result;
        f.CallPacked(args, n_args, &result);
        TVMFFIEnvSetStream(kDLCUDA, device, original, nullptr);
    };

    // ------------------------------------------------ driver API, no FFI --
    CUmodule cu_module;
    CUfunction cu_add;
    CHECK_CU(cuModuleLoad(&cu_module, "bench_native_add.cubin"));
    CHECK_CU(cuModuleGetFunction(&cu_add, cu_module, "native_add"));
    int const n32    = static_cast<int>(n);
    int const blocks = (n32 + 255) / 256;

    // ---------------------------------------------------- CUDA graph replay --
    // Capture on a non-default stream, which capture requires. Deferred until
    // after the warmup below: the exported host shim loads its cubin lazily on
    // first call, and doing that inside a capture touches the legacy default
    // stream and fails with STREAM_CAPTURE_IMPLICIT.
    cudaStream_t capture_stream;
    CHECK_CUDA(cudaStreamCreate(&capture_stream));
    cudaGraph_t captured;
    cudaGraphExec_t replay = nullptr;
    // NOTE: what gets captured here is the NATIVE kernel, not the CuTeDSL one.
    // Capturing the exported host shim fails with cudaErrorStreamCaptureImplicit
    // (906) in every capture mode -- it touches the legacy default stream
    // somewhere inside. That is worth knowing on its own (a CuTeDSL kernel is
    // not capturable as shipped), but it does not change this row: replay cost
    // is a property of the graph launch, not of which kernel is in the graph.
    // One node either way.
    auto capture_native_call = [&](int nodes, cudaGraphExec_t *out) -> cudaError_t {
        cudaError_t rc = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
        if (rc != cudaSuccess) return rc;
        for (int i = 0; i < nodes; i++) native_add_launch(da, db, dc, static_cast<int>(n), capture_stream);
        rc = cudaStreamEndCapture(capture_stream, &captured);
        if (rc != cudaSuccess) return rc;
        return cudaGraphInstantiate(out, captured, nullptr, nullptr, 0);
    };

    // Replaying a ONE-node graph saves almost nothing: the per-launch cost is
    // mostly the submit itself, and a graph still has to be submitted. The
    // often-quoted ~0.1 µs is a per-KERNEL figure from a graph holding many of
    // them, so both are measured -- a 1-node replay, and a 32-node replay
    // divided by 32.
    constexpr int GRAPH_NODES = 32;
    cudaGraphExec_t replay_many = nullptr;

    // ----------------------------------------------------------------- arms --
    std::vector<Arm> arms;
    arms.push_back({"ffi_module    (cutlass AOT .so, C++, TVM FFI)", [&] { call_ffi(fn_from_module); }, {}});
    arms.push_back({"ffi_global    (cutlass AOT, C++, global table)", [&] { call_ffi(fn_from_global); }, {}});
    arms.push_back({"fe_container  (FE graph.execute, from container)",
                    [&] { (void)from_container.execute(handle, ptrs.data(), (int)ptrs.size(), workspace); },
                    {}});
    arms.push_back({"fe_global     (FE graph.execute, from registry)",
                    [&] { (void)from_registry.execute(handle, ptrs.data(), (int)ptrs.size(), workspace); },
                    {}});
    arms.push_back({"native_launch (hand-written, cudaLaunchKernel)",
                    [&] { native_add_launch(da, db, dc, n32, stream); },
                    {}});
    arms.push_back({"raw_cubin     (cuLaunchKernel, no FFI)",
                    [&] {
                        void *params[] = {&da, &db, &dc, (void *)&n32};
                        cuLaunchKernel(cu_add, blocks, 1, 1, 256, 1, 1, 0, (CUstream)stream, params, nullptr);
                    },
                    {}});
    arms.push_back({"graph_replay  (cudaGraph replay, 1 node)",
                    [&] {
                        if (replay) (void)cudaGraphLaunch(replay, capture_stream);
                    },
                    {}});
    arms.push_back({"graph_replay32(cudaGraph replay, 32 nodes)",
                    [&] {
                        if (replay_many) (void)cudaGraphLaunch(replay_many, capture_stream);
                    },
                    {}});
    // Repeat of arm 0. Its difference from arm 0 is the noise floor: any gap
    // between other arms smaller than this is not a measurement.
    arms.push_back({"control       (= ffi_module, repeated)", [&] { call_ffi(fn_from_module); }, {}});

    constexpr int WARMUP = 300;
    constexpr int BURSTS = 41;  // longer runs get queue-throttled and stop measuring host cost
    constexpr int CALLS  = 100;

    for (int i = 0; i < WARMUP; i++) {
        for (auto &arm : arms) arm.call();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Everything is loaded now, so the capture sees only the launch.
    CHECK_CUDA(capture_native_call(1, &replay));
    CHECK_CUDA(capture_native_call(GRAPH_NODES, &replay_many));
    for (int i = 0; i < WARMUP; i++) {
        arms[6].call();  // warm the replay arms too
        arms[7].call();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto burst = [&](Arm &arm) {
        auto const t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < CALLS; i++) arm.call();
        auto const t1 = std::chrono::steady_clock::now();
        cudaDeviceSynchronize();  // drain BETWEEN bursts, outside the timing
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / CALLS;
    };

    for (int b = 0; b < BURSTS; b++) {
        for (auto &arm : arms) arm.samples.push_back(burst(arm));
    }

    std::printf("kernel %s, n=%lld, %d bursts x %d calls, round-robin, medians\n\n",
                kernel.c_str(),
                static_cast<long long>(n),
                BURSTS,
                CALLS);
    std::printf("  %-46s %10s\n", "call path", "us/call");
    double const floor_us = median(arms[4].samples);
    for (auto &arm : arms) {
        double const us = median(arm.samples);
        std::printf("  %-46s %10.3f", arm.label, us);
        if (&arm != &arms[4]) std::printf("   (%+.3f vs native_launch)", us - floor_us);
        std::printf("\n");
    }
    std::printf("  %-46s %10.3f   (32-node replay / 32)\n",
                "  ^ per kernel in the 32-node graph",
                median(arms[7].samples) / GRAPH_NODES);
    std::printf("\n  front door over an OSS-engine plan: fe_container - ffi_module = %+.3f us\n",
                median(arms[2].samples) - median(arms[0].samples));

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    if (workspace) cudaFree(workspace);
    cudnnDestroy(handle);
    return 0;
}
