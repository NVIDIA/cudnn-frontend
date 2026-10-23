// Complete standalone C++ sample.
// API page: docs/api/cpp/add.md
// API: add
// Use case: Basic
// Source: adapted from `python/pygraph/pointwise.cpp` and `samples/cpp/misc/pointwise.cpp`.

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
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");

    cudaFree(workspace);
    cudnnDestroy(handle);
}

int
main() {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    constexpr int64_t n = 24;
    float* A_dev        = managed<float>(n);
    float* B_dev        = managed<float>(n);
    float* Y_dev        = managed<float>(n);
    for (int64_t i = 0; i < n; ++i) A_dev[i] = dis(gen);
    for (int64_t i = 0; i < n; ++i) B_dev[i] = dis(gen);
    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({2, 3, 4})
                              .set_stride({12, 4, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto B = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({2, 3, 4})
                              .set_stride({12, 4, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto Y = graph.pointwise(A,
                             B,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    Y->set_output(true).set_name("Y").set_data_type(fe::DataType_t::FLOAT);
    VariantPack variant_pack = {{A, A_dev}, {B, B_dev}, {Y, Y_dev}};
    build_and_execute(graph, variant_pack);
    for (int64_t i = 0; i < n; ++i) {
        float actual   = Y_dev[i];
        float expected = A_dev[i] + B_dev[i];
        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            std::fabs(actual - expected) > 1.0e-5f) {
            throw std::runtime_error("add result mismatch");
        }
    }
    std::cout << "add result matches reference\n";
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(Y_dev);
    return 0;
}
