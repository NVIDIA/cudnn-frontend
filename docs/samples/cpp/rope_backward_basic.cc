// Complete standalone C++ sample.
// API page: docs/api/cpp/rope_backward.md
// API: rope_backward
// Use case: Basic
// Source: adapted from `test/python/test_oss_rope.py` and the C++ graph API.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace fe = cudnn_frontend;

void
check(fe::error_t status) {
    if (!status.is_good()) throw std::runtime_error(status.get_message());
}

template <typename T>
T*
managed(int64_t count) {
    T* ptr = nullptr;
    cudaMallocManaged(&ptr, count * sizeof(T));
    return ptr;
}

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;

void
build_and_execute(fe::graph::Graph& graph, VariantPack& variant_pack) {
    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);
    check(graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));
    void* workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    check(graph.execute(handle, variant_pack, workspace));
    cudaDeviceSynchronize();
    cudaFree(workspace);
    cudnnDestroy(handle);
}

int
main() {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    int64_t const B = 1, H = 2, S = 16, D = 64;
    nv_bfloat16* X_dev = managed<nv_bfloat16>(B * H * S * D);
    float* F_dev       = managed<float>(S * D);
    nv_bfloat16* Y_dev = managed<nv_bfloat16>(B * H * S * D);
    for (int64_t i = 0; i < (B * H * S * D); ++i) X_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t i = 0; i < (S * D); ++i) F_dev[i] = dis(gen);
    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({B, H, S, D})
                              .set_stride({H * S * D, S * D, D, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto F = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("freqs")
                              .set_dim({S, 1, 1, D})
                              .set_stride({D, D, D, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto Y =
        graph.rope_backward(X, F, fe::graph::RoPE_backward_attributes().set_name("rope_backward"));
    Y->set_output(true)
        .set_name("Y")
        .set_dim({B, H, S, D})
        .set_stride({H * S * D, S * D, D, 1})
        .set_data_type(fe::DataType_t::BFLOAT16);
    VariantPack variant_pack = {{X, X_dev}, {F, F_dev}, {Y, Y_dev}};
    build_and_execute(graph, variant_pack);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t s = 0; s < S; ++s) {
                for (int64_t j = 0; j < D / 2; ++j) {
                    auto base      = ((b * H + h) * S + s) * D;
                    auto dy1       = __bfloat162float(X_dev[base + j]);
                    auto dy2       = __bfloat162float(X_dev[base + D / 2 + j]);
                    auto angle     = F_dev[s * D + j];
                    auto expected1 = dy1 * std::cos(angle) + dy2 * std::sin(angle);
                    auto expected2 = dy2 * std::cos(angle) - dy1 * std::sin(angle);
                    auto actual1   = __bfloat162float(Y_dev[base + j]);
                    auto actual2   = __bfloat162float(Y_dev[base + D / 2 + j]);
                    if (!std::isfinite(actual1) || !std::isfinite(actual2) ||
                        !std::isfinite(expected1) || !std::isfinite(expected2) ||
                        std::fabs(actual1 - expected1) > 3.0e-2f ||
                        std::fabs(actual2 - expected2) > 3.0e-2f) {
                        throw std::runtime_error("rope backward result mismatch");
                    }
                }
            }
        }
    }
    std::cout << "rope backward matches reference\n";
    cudaFree(X_dev);
    cudaFree(F_dev);
    cudaFree(Y_dev);
    return 0;
}
