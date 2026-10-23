// Complete standalone C++ sample.
// API page: docs/api/cpp/layernorm.md
// API: layernorm
// Use case: Basic
// Source: adapted from `samples/cpp/norm/layernorm.cpp`.

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
    int64_t const batch_size  = 8;
    int64_t const hidden_size = 64;
    int64_t const x_count     = batch_size * hidden_size;
    int64_t const stat_count  = batch_size;
    nv_bfloat16* X_dev        = managed<nv_bfloat16>(x_count);
    nv_bfloat16* Y_dev        = managed<nv_bfloat16>(x_count);
    nv_bfloat16* Scale_dev    = managed<nv_bfloat16>(hidden_size);
    nv_bfloat16* Bias_dev     = managed<nv_bfloat16>(hidden_size);
    float* Mean_dev           = managed<float>(stat_count);
    float* InvVar_dev         = managed<float>(stat_count);
    for (int64_t i = 0; i < x_count; ++i) X_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t i = 0; i < hidden_size; ++i) Scale_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t i = 0; i < hidden_size; ++i) Bias_dev[i] = __float2bfloat16(dis(gen));

    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X       = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch_size, hidden_size, 1, 1})
                              .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto Scale   = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, hidden_size, 1, 1})
                                  .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto Bias    = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, hidden_size, 1, 1})
                                 .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto Epsilon = graph.tensor(1.0e-3f);
    auto attrs   = fe::graph::Layernorm_attributes()
                     .set_name("layernorm")
                     .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                     .set_epsilon(Epsilon)
                     .set_compute_data_type(fe::DataType_t::FLOAT);
    auto [Y, Mean, InvVar] = graph.layernorm(X, Scale, Bias, attrs);
    Y->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);
    Mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    InvVar->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    VariantPack variant_pack = {{X, X_dev},
                                {Scale, Scale_dev},
                                {Bias, Bias_dev},
                                {Y, Y_dev},
                                {Mean, Mean_dev},
                                {InvVar, InvVar_dev}};
    build_and_execute(graph, variant_pack);
    for (int64_t row = 0; row < stat_count; ++row) {
        float mean = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j)
            mean += __bfloat162float(X_dev[row * hidden_size + j]);
        mean /= static_cast<float>(hidden_size);
        float variance = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j) {
            auto delta = __bfloat162float(X_dev[row * hidden_size + j]) - mean;
            variance += delta * delta;
        }
        auto inv_variance = 1.0f / std::sqrt(variance / static_cast<float>(hidden_size) + 1.0e-3f);
        auto actual_mean  = Mean_dev[row];
        auto actual_inv_variance = InvVar_dev[row];
        if (!std::isfinite(actual_mean) || !std::isfinite(actual_inv_variance) ||
            !std::isfinite(mean) || !std::isfinite(inv_variance) ||
            std::fabs(actual_mean - mean) > 1.0e-4f ||
            std::fabs(actual_inv_variance - inv_variance) > 1.0e-3f) {
            throw std::runtime_error("layernorm statistics mismatch");
        }
        for (int64_t j = 0; j < hidden_size; ++j) {
            auto expected = (__bfloat162float(X_dev[row * hidden_size + j]) - mean) * inv_variance *
                                __bfloat162float(Scale_dev[j]) +
                            __bfloat162float(Bias_dev[j]);
            auto actual = __bfloat162float(Y_dev[row * hidden_size + j]);
            if (!std::isfinite(actual) || !std::isfinite(expected) ||
                std::fabs(actual - expected) > 1.25e-1f) {
                throw std::runtime_error("layernorm output mismatch");
            }
        }
    }
    std::cout << "layernorm output and statistics match reference\n";
    cudaFree(X_dev);
    cudaFree(Y_dev);
    cudaFree(Scale_dev);
    cudaFree(Bias_dev);
    cudaFree(Mean_dev);
    cudaFree(InvVar_dev);
    return 0;
}
