// Complete standalone C++ sample.
// API page: docs/api/cpp/matmul.md
// API: matmul
// Use case: Basic
// Source: adapted from `samples/cpp/matmul/matmuls.cpp`.

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
    int64_t const b    = 2;
    int64_t const m    = 4;
    int64_t const k    = 16;
    int64_t const n    = 8;
    nv_bfloat16* A_dev = managed<nv_bfloat16>(b * m * k);
    nv_bfloat16* B_dev = managed<nv_bfloat16>(k * n);
    nv_bfloat16* C_dev = managed<nv_bfloat16>(b * m * n);
    for (int64_t i = 0; i < b * m * k; ++i) A_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t i = 0; i < k * n; ++i) B_dev[i] = __float2bfloat16(dis(gen));
    fe::graph::Graph graph{};
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({b, m, k})
                              .set_stride({m * k, k, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto B = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({1, k, n})
                              .set_stride({k * n, n, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto C = graph.matmul(A,
                          B,
                          fe::graph::Matmul_attributes().set_name("matmul").set_compute_data_type(
                              fe::DataType_t::FLOAT));
    C->set_output(true)
        .set_name("C")
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_dim({b, m, n})
        .set_stride({m * n, n, 1});
    VariantPack variant_pack = {{A, A_dev}, {B, B_dev}, {C, C_dev}};
    build_and_execute(graph, variant_pack);
    for (int64_t batch = 0; batch < b; ++batch) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                float expected = 0.0f;
                for (int64_t inner = 0; inner < k; ++inner) {
                    expected += __bfloat162float(A_dev[(batch * m + row) * k + inner]) *
                                __bfloat162float(B_dev[inner * n + col]);
                }
                auto actual = __bfloat162float(C_dev[(batch * m + row) * n + col]);
                if (!std::isfinite(actual) || !std::isfinite(expected) ||
                    std::fabs(actual - expected) > 1.0e-2f) {
                    throw std::runtime_error("matmul result mismatch");
                }
            }
        }
    }
    std::cout << "matmul result matches reference\n";
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    return 0;
}
