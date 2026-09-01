// Host cost of an SDPA forward launch, two call paths, one process.
//
//   backend   the cuDNN backend engine, graph built and executed in C++
//   aot       the FROST SDPA kernel, compiled in Python, executed from a
//             container this process opened -- no Python, no cutlass, no JIT
//
// The two are NOT the same kernel: the backend engine has no AOT export, so the
// artifact necessarily carries the other SDPA implementation. Same problem,
// same buffers, same clock -- what is being compared is the CALL PATH, which is
// what host overhead is.
//
// Method follows bench_cpu_costs.cpp: submit time, no synchronization inside a
// burst, arms round-robin within each burst so drift hits both equally, median
// over bursts, and the queue drained BETWEEN bursts outside the timed region.
// A repeated arm gives the noise floor.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cudnn_frontend.h>
#include <cudnn_frontend/kernel_library.h>

namespace fe = cudnn_frontend;

// FE is built here in dynamic-loading mode (as its own python module is), so
// one translation unit has to own the handle and open libcudnn.
namespace cudnn_frontend {
void *cudnn_dlhandle = nullptr;
}

namespace {

#define CHECK_CUDA(expr)                                                                                    \
    do {                                                                                                    \
        cudaError_t const _e = (expr);                                                                      \
        if (_e != cudaSuccess) {                                                                            \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(_e));    \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

#define CHECK_FE(expr)                                                                    \
    do {                                                                                  \
        auto _s = (expr);                                                           \
        if (_s.is_bad()) {                                                                \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, _s.get_message().c_str()); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

double
median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Minimal scrape of the bench manifest; the container carries the real contract.
int64_t
manifest_int(std::string const &text, char const *key) {
    std::string const pat = std::string("\"") + key + "\":";
    size_t p              = text.find(pat);
    if (p == std::string::npos) return -1;
    return std::strtoll(text.c_str() + p + pat.size(), nullptr, 10);
}

struct Arm {
    char const *label;
    std::function<void()> call;
    std::vector<double> samples;
};

constexpr int64_t Q_UID = 101, K_UID = 102, V_UID = 103, O_UID = 104;

}  // namespace

int
main(int argc, char **argv) {
    std::string const container = (argc > 1) ? argv[1] : "sdpa.cudnn";

    std::ifstream mf(container + ".bench.json");
    if (!mf.good()) {
        std::fprintf(stderr, "cannot read %s.bench.json -- run bench_sdpa_export.py first\n", container.c_str());
        return 1;
    }
    std::string const bench((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    int64_t const B = manifest_int(bench, "dims");  // first element of dims
    (void)B;

    // Must match bench_sdpa_export.py.
    int64_t const b = 2, h = 8, s = 1024, d = 128;
    std::vector<int64_t> const dim{b, h, s, d};
    std::vector<int64_t> const stride{s * h * d, d, h * d, 1};
    int64_t const elems = b * h * s * d;

    fe::cudnn_dlhandle = ::dlopen("libcudnn.so", RTLD_NOW | RTLD_GLOBAL);
    if (fe::cudnn_dlhandle == nullptr) {
        std::fprintf(stderr, "dlopen(libcudnn.so): %s\n", ::dlerror());
        return 1;
    }

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "could not create a cuDNN handle\n");
        return 1;
    }
    cudaStream_t stream = nullptr;
    cudnnGetStream(handle, &stream);

    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    CHECK_CUDA(cudaMalloc(&dq, elems * 2));
    CHECK_CUDA(cudaMalloc(&dk, elems * 2));
    CHECK_CUDA(cudaMalloc(&dv, elems * 2));
    CHECK_CUDA(cudaMalloc(&dout, elems * 2));
    CHECK_CUDA(cudaMemset(dq, 0, elems * 2));
    CHECK_CUDA(cudaMemset(dk, 0, elems * 2));
    CHECK_CUDA(cudaMemset(dv, 0, elems * 2));

    // ------------------------------------------------------- backend arm --
    // The ordinary C++ lifecycle: build here, execute here. This is the path a
    // C++ integration takes today.
    fe::graph::Graph backend;
    backend.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = backend.tensor(fe::graph::Tensor_attributes().set_name("q").set_uid(Q_UID).set_dim(dim).set_stride(stride));
    auto K = backend.tensor(fe::graph::Tensor_attributes().set_name("k").set_uid(K_UID).set_dim(dim).set_stride(stride));
    auto V = backend.tensor(fe::graph::Tensor_attributes().set_name("v").set_uid(V_UID).set_dim(dim).set_stride(stride));

    auto opts = fe::graph::SDPA_attributes()
                    .set_name("sdpa")
                    .set_generate_stats(false)
                    .set_attn_scale(1.0f / 11.313708f)  // 1/sqrt(128)
                    .set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                    .set_diagonal_band_right_bound(0);

    auto [O, Stats] = backend.sdpa(Q, K, V, opts);
    (void)Stats;
    O->set_output(true).set_dim(dim).set_stride(stride).set_uid(O_UID);

    CHECK_FE(backend.validate());
    CHECK_FE(backend.build_operation_graph(handle));
    CHECK_FE(backend.create_execution_plans({fe::HeurMode_t::A}));
    CHECK_FE(backend.check_support(handle));
    CHECK_FE(backend.build_plans(handle));

    int64_t backend_ws = 0;
    CHECK_FE(backend.get_workspace_size(backend_ws));
    void *backend_workspace = nullptr;
    if (backend_ws > 0) CHECK_CUDA(cudaMalloc(&backend_workspace, backend_ws));

    std::unordered_map<int64_t, void *> pack{{Q_UID, dq}, {K_UID, dk}, {V_UID, dv}, {O_UID, dout}};

    // ----------------------------------------------------------- aot arm --
    fe::KernelLibrary lib;
    CHECK_FE(fe::import_from_disk(container, &lib, handle));
    fe::graph::Graph aot;
    CHECK_FE(lib.get("sdpa_fwd_causal_f16", &aot));

    std::vector<int64_t> const order = aot.get_variant_pack_uids_sorted();
    std::vector<void *> ptrs(order.size());
    {
        // The gather a caller does per call; the ORDER is resolved once, here.
        std::unordered_map<int64_t, void *> by_uid;
        // uids come from the exporter's manifest, which lists them by role.
        std::string const uids = bench.substr(bench.find("\"uids\""));
        int64_t const uq = manifest_int(uids, "q"), uk = manifest_int(uids, "k");
        int64_t const uv = manifest_int(uids, "v"), uo = manifest_int(uids, "o");
        by_uid[uq] = dq;
        by_uid[uk] = dk;
        by_uid[uv] = dv;
        by_uid[uo] = dout;
        for (size_t i = 0; i < order.size(); i++) {
            auto it = by_uid.find(order[i]);
            if (it == by_uid.end()) {
                std::fprintf(stderr, "manifest has no buffer for uid %lld\n", (long long)order[i]);
                return 1;
            }
            ptrs[i] = it->second;
        }
    }

    int64_t aot_ws = 0;
    CHECK_FE(aot.get_workspace_size(aot_ws));
    void *aot_workspace = nullptr;
    if (aot_ws > 0) CHECK_CUDA(cudaMalloc(&aot_workspace, aot_ws));

    // ---------------------------------------------------------------- arms --
    std::vector<Arm> arms;
    arms.push_back({"backend  (cuDNN backend engine, C++ build+execute)",
                    [&] { (void)backend.execute(handle, pack, backend_workspace); },
                    {}});
    arms.push_back({"aot      (python compile, C++ execute from container)",
                    [&] { (void)aot.execute(handle, ptrs.data(), (int)ptrs.size(), aot_workspace); },
                    {}});
    // Diagnostic: the three cudaMemsetAsync the artifact issues to zero the
    // dummy buffers it carved into the workspace. Not a call path -- it isolates
    // how much of the aot arm is those memsets rather than the launch.
    arms.push_back({"  (diag) 3x cudaMemsetAsync only",
                    [&] {
                        cudaMemsetAsync(static_cast<char *>(aot_workspace) + 65536, 0, 32, stream);
                        cudaMemsetAsync(static_cast<char *>(aot_workspace) + 65664, 0, 8, stream);
                        cudaMemsetAsync(static_cast<char *>(aot_workspace) + 65792, 0, 8, stream);
                    },
                    {}});
    // Repeat of arm 0: its gap from arm 0 is the noise floor. Any difference
    // between the two paths smaller than this is not a measurement.
    arms.push_back({"control  (= backend, repeated)",
                    [&] { (void)backend.execute(handle, pack, backend_workspace); },
                    {}});

    constexpr int WARMUP = 300;
    constexpr int BURSTS = 41;
    constexpr int CALLS  = 100;

    for (int i = 0; i < WARMUP; i++)
        for (auto &arm : arms) arm.call();
    CHECK_CUDA(cudaDeviceSynchronize());

    auto burst = [&](Arm &arm) {
        auto const t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < CALLS; i++) arm.call();
        auto const t1 = std::chrono::steady_clock::now();
        cudaDeviceSynchronize();  // drained BETWEEN bursts, outside the timing
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / CALLS;
    };

    for (int burst_i = 0; burst_i < BURSTS; burst_i++)
        for (auto &arm : arms) arm.samples.push_back(burst(arm));

    std::printf("SDPA forward, causal, B=%lld H=%lld S=%lld D=%lld, fp16\n", (long long)b, (long long)h, (long long)s, (long long)d);
    std::printf("%d bursts x %d calls, round-robin, medians\n\n", BURSTS, CALLS);
    std::printf("  %-52s %10s\n", "call path", "us/call");
    for (auto &arm : arms) std::printf("  %-52s %10.3f\n", arm.label, median(arm.samples));

    // Submit time only measures the HOST while the queue has room. If the GPU
    // is slower than the submit loop the queue fills and submit time decays
    // into device time, which is a throughput number wearing a latency costume.
    // So measure the device side too and say which clock is binding.
    std::printf("\n  %-52s %10s %10s\n", "", "submit", "wall/sync");
    for (size_t ai = 0; ai < 2; ai++) {
        std::vector<double> wall;
        for (int b_i = 0; b_i < 11; b_i++) {
            CHECK_CUDA(cudaDeviceSynchronize());
            auto const t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < CALLS; i++) arms[ai].call();
            CHECK_CUDA(cudaDeviceSynchronize());  // INSIDE the timed region
            auto const t1 = std::chrono::steady_clock::now();
            wall.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / CALLS);
        }
        double const sub = median(arms[ai].samples), w = median(wall);
        std::printf("  %-52s %10.3f %10.3f   %s\n",
                    arms[ai].label,
                    sub,
                    w,
                    (w > sub * 1.15) ? "host-bound" : "DEVICE-BOUND: submit is not host cost");
    }

    double const noise = median(arms[3].samples) - median(arms[0].samples);
    std::printf("\n  noise floor (control - backend)      %+.3f us\n", noise);
    std::printf("  aot - backend                        %+.3f us\n", median(arms[1].samples) - median(arms[0].samples));
    std::printf("  workspace: backend %lld B, aot %lld B\n", (long long)backend_ws, (long long)aot_ws);

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout);
    if (backend_workspace) cudaFree(backend_workspace);
    if (aot_workspace) cudaFree(aot_workspace);
    cudnnDestroy(handle);
    return 0;
}
