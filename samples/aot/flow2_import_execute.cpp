// Flow 2, deploy box (C++): open the container and run every kernel in it.
//
// The same job as flow2_import_execute.py, with no Python in the process at
// all. This binary links cudnn-frontend and libtvm_ffi and nothing else -- no
// cutlass, no NVRTC, no kernel toolchain.
//
//     ./flow2_import_execute [mykernels.cudnn]

#include <cudnn_frontend/kernel_library.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace fe = cudnn_frontend;

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

namespace {

// The manifest is a convenience for the samples, not part of the contract: the
// uid ORDER comes from the container via get_variant_pack_uids_sorted(); this
// only says which uid is A / B / C and how big the buffers are, which a real
// caller knows from its own model. Scanned by hand so the sample grows no JSON
// dependency of its own.
long long
manifest_int(std::string const &text, size_t from, char const *key) {
    auto const at = text.find(std::string("\"") + key + "\"", from);
    if (at == std::string::npos) return -1;
    return std::strtoll(text.c_str() + text.find(':', at) + 1, nullptr, 10);
}

long long
manifest_elems(std::string const &text, size_t from) {
    auto const open  = text.find('[', text.find("\"shape\"", from));
    auto const close = text.find(']', open);
    long long n      = 1;
    for (size_t i = open + 1; i < close; i++) {
        if (std::isdigit(static_cast<unsigned char>(text[i]))) {
            n *= std::strtoll(text.c_str() + i, nullptr, 10);
            while (i < close && std::isdigit(static_cast<unsigned char>(text[i]))) i++;
        }
    }
    return n;
}

}  // namespace

int
main(int argc, char **argv) {
    std::string const container = (argc > 1) ? argv[1] : "mykernels.cudnn";

    std::ifstream mf(container + ".manifest.json");
    if (!mf.good()) {
        std::fprintf(stderr, "cannot read %s.manifest.json -- run flow2_export.py first\n", container.c_str());
        return 1;
    }
    std::string const manifest((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "could not create a cuDNN handle\n");
        return 1;
    }

    // Version, architecture and missing-runtime-dependency mismatches are all
    // rejected HERE, loudly. A misread artifact is never an illegal access.
    fe::KernelLibrary lib;
    CHECK_FE(fe::import_from_disk(container, &lib, handle));

    std::printf("opened %s holding %zu kernel(s):", container.c_str(), lib.size());
    for (auto const &name : lib.keys()) std::printf(" %s", name.c_str());
    std::printf("\n");

    int failures = 0;
    for (auto const &name : lib.keys()) {
        fe::graph::Graph graph;
        CHECK_FE(lib.get(name, &graph));  // BY NAME, never by position

        // Resolved once, at startup.
        auto const uid_order   = graph.get_variant_pack_uids_sorted();
        int64_t workspace_size = 0;
        CHECK_FE(graph.get_workspace_size(workspace_size));

        auto const at         = manifest.find("\"" + name + "\"");
        long long const n     = manifest_elems(manifest, at);
        long long const a_uid = manifest_int(manifest, at, "a_uid");
        long long const b_uid = manifest_int(manifest, at, "b_uid");

        float *da, *db, *dc;
        CHECK_CUDA(cudaMalloc(&da, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&db, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dc, n * sizeof(float)));
        void *workspace = nullptr;
        if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

        std::vector<float> ha(n), hb(n), hc(n);
        for (long long i = 0; i < n; i++) {
            ha[i] = static_cast<float>(i % 97) * 0.5f;
            hb[i] = static_cast<float>(i % 31) * 0.25f;
        }
        CHECK_CUDA(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dc, 0xff, n * sizeof(float)));  // poison, so a no-op is visible

        // Hot path: gather in uid_order, then execute.
        std::vector<void *> ptrs(uid_order.size());
        for (size_t i = 0; i < uid_order.size(); i++) {
            ptrs[i] = (uid_order[i] == a_uid)   ? static_cast<void *>(da)
                      : (uid_order[i] == b_uid) ? static_cast<void *>(db)
                                                : static_cast<void *>(dc);
        }
        CHECK_FE(graph.execute(handle, ptrs.data(), static_cast<int>(ptrs.size()), workspace));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (long long i = 0; i < n; i++) {
            max_err = std::max(max_err, static_cast<double>(std::fabs(hc[i] - (ha[i] + hb[i]))));
        }
        std::printf(
            "  %-16s n=%-8lld max |err| = %g%s\n", name.c_str(), n, max_err, max_err == 0.0 ? "" : "   <-- MISMATCH");
        if (max_err != 0.0) failures++;

        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        if (workspace) cudaFree(workspace);
    }

    cudnnDestroy(handle);
    return failures ? 1 : 0;
}
