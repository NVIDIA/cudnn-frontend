/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

// Runs a plan that Python compiled ahead of time, from C++ with no Python, no
// JIT and no cuDNN handle.
//
//     run_plan <plan.bin> <dir>
//
// <plan.bin> is graph.serialize() of a built graph whose selected plan is a
// CuTeDSL engine (samples/aot/export_plan.py writes one). For every tensor the
// plan binds, <dir>/<uid>.bin holds its bytes; they are uploaded, the plan runs
// once, and every buffer is written back to its file.
//
// Build (header-only frontend; cuDNN itself is never loaded by an AOT plan):
//     g++ -std=c++17 -DNV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING -I include -I $CUDA/include -I $CUDNN/include
//         samples/aot/run_plan.cpp -L $CUDA/lib64 -lcudart -ldl -o run_plan
// Run with libtvm_ffi.so and libcute_dsl_runtime.so on LD_LIBRARY_PATH.

#include <cudnn_frontend.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

#ifdef NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
// The frontend's cuDNN handle when cuDNN is loaded at runtime; an AOT plan never loads it.
void *cudnn_frontend::cudnn_dlhandle = nullptr;
#endif

static std::vector<uint8_t>
read_file(std::string const &path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

#define CHECK_CUDA(x)                                                            \
    do {                                                                         \
        cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(e_)); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_FE(x)                                                                \
    do {                                                                           \
        auto s_ = (x);                                                             \
        if (s_.is_bad()) {                                                         \
            std::fprintf(stderr, "%s failed: %s\n", #x, s_.get_message().c_str()); \
            return 1;                                                              \
        }                                                                          \
    } while (0)

int
main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <plan.bin> <dir>\n", argv[0]);
        return 2;
    }
    std::string const dir = argv[2];
    CHECK_CUDA(cudaSetDevice(0));

    fe::graph::Graph graph;
    CHECK_FE(graph.deserialize(read_file(argv[1])));  // no handle: an AOT plan never calls cuDNN

    int64_t workspace_size = 0;
    CHECK_FE(graph.get_workspace_size(workspace_size));
    void *workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size > 0 ? workspace_size : 1));

    std::unordered_map<int64_t, void *> pack;
    std::unordered_map<int64_t, size_t> sizes;
    for (int64_t uid : graph.get_variant_pack_uids_sorted()) {
        auto const bytes = read_file(dir + "/" + std::to_string(uid) + ".bin");
        void *ptr        = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, bytes.size()));
        CHECK_CUDA(cudaMemcpy(ptr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        pack[uid]  = ptr;
        sizes[uid] = bytes.size();
    }

    CHECK_FE(graph.execute(nullptr, pack, workspace));
    CHECK_CUDA(cudaDeviceSynchronize());

    for (auto const &[uid, ptr] : pack) {
        std::vector<uint8_t> bytes(sizes[uid]);
        CHECK_CUDA(cudaMemcpy(bytes.data(), ptr, bytes.size(), cudaMemcpyDeviceToHost));
        std::ofstream(dir + "/" + std::to_string(uid) + ".bin", std::ios::binary)
            .write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        CHECK_CUDA(cudaFree(ptr));
    }
    CHECK_CUDA(cudaFree(workspace));
    std::printf("ran %zu tensors, %lld-byte workspace\n", pack.size(), static_cast<long long>(workspace_size));
    return 0;
}
