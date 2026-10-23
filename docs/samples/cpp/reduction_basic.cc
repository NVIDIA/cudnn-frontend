// Complete standalone C++ sample.
// API page: docs/api/cpp/reduction.md
// API: reduction
// Use case: Basic
// Source: adapted from `samples/cpp/convolution/fp8_fprop.cpp`.

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cmath>
#include <cstdint>
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
    constexpr int64_t input_count  = 10 * 8 * 4 * 4;
    constexpr int64_t output_count = 10 * 4 * 4;
    float* X_dev                   = managed<float>(input_count);
    float* Y_dev                   = managed<float>(output_count);
    std::mt19937 generator(1234);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (int64_t i = 0; i < input_count; ++i) X_dev[i] = distribution(generator);
    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({10, 8, 4, 4})
                              .set_stride({128, 1, 32, 8})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto Y = graph.reduction(X,
                             fe::graph::Reduction_attributes()
                                 .set_name("reduction")
                                 .set_mode(fe::ReductionMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    Y->set_output(true)
        .set_name("Y")
        .set_dim({10, 1, 4, 4})
        .set_stride({16, 16, 4, 1})
        .set_data_type(fe::DataType_t::FLOAT);
    VariantPack variant_pack = {{X, X_dev}, {Y, Y_dev}};
    build_and_execute(graph, variant_pack);

    for (int64_t batch = 0; batch < 10; ++batch) {
        for (int64_t height = 0; height < 4; ++height) {
            for (int64_t width = 0; width < 4; ++width) {
                float expected = 0.0f;
                for (int64_t channel = 0; channel < 8; ++channel) {
                    expected += X_dev[batch * 128 + channel + height * 32 + width * 8];
                }
                auto output_index = batch * 16 + height * 4 + width;
                if (std::abs(Y_dev[output_index] - expected) > 1.0e-5f)
                    throw std::runtime_error("reduction result mismatch");
            }
        }
    }
    std::cout << "reduction output: shape=[10, 1, 4, 4], values=[" << Y_dev[0] << ", " << Y_dev[1]
              << ", " << Y_dev[2] << ", " << Y_dev[3] << ", ...]\n";
    cudaFree(X_dev);
    cudaFree(Y_dev);
    return 0;
}
