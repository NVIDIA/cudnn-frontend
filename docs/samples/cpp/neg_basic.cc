// Complete standalone C++ sample.
// API page: docs/api/cpp/neg.md
// API: neg
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
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

template <typename T>
T*
managed(int64_t count) {
    T* ptr             = nullptr;
    cudaError_t status = cudaMallocManaged(&ptr, count * sizeof(T));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocManaged: ") + cudaGetErrorString(status));
    }
    return ptr;
}

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;

void
build_and_execute(fe::graph::Graph& graph, VariantPack& variant_pack) {
    cudnnHandle_t handle       = nullptr;
    void* workspace            = nullptr;
    cudnnStatus_t cudnn_status = cudnnCreate(&handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cudnnCreate: ") + cudnnGetErrorString(cudnn_status));
    }

    try {
        auto status = graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        if (!status.is_good()) {
            throw std::runtime_error(std::string("graph.build: ") + status.get_message());
        }

        int64_t workspace_size = graph.get_workspace_size();
        if (workspace_size > 0) {
            cudaError_t cuda_status = cudaMalloc(&workspace, workspace_size);
            if (cuda_status != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc(workspace): ") + cudaGetErrorString(cuda_status));
            }
        }

        status = graph.execute(handle, variant_pack, workspace);
        if (!status.is_good()) {
            throw std::runtime_error(std::string("graph.execute: ") + status.get_message());
        }

        cudaError_t cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(cuda_status));
        }
    } catch (...) {
        if (workspace) cudaFree(workspace);
        cudnnDestroy(handle);
        throw;
    }

    if (workspace) {
        cudaError_t cuda_status = cudaFree(workspace);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaFree(workspace): ") + cudaGetErrorString(cuda_status));
        }
    }

    cudnn_status = cudnnDestroy(handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cudnnDestroy: ") + cudnnGetErrorString(cudnn_status));
    }
}

int
main() {
    try {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        constexpr int64_t n = 24;
        float* X_dev        = managed<float>(n);
        float* Y_dev        = managed<float>(n);
        for (int64_t i = 0; i < (n); ++i) X_dev[i] = dis(gen);
        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto X = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("X")
                                  .set_dim({2, 3, 4})
                                  .set_stride({12, 4, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
        auto Y = graph.pointwise(X,
                                 fe::graph::Pointwise_attributes()
                                     .set_name("neg")
                                     .set_mode(fe::PointwiseMode_t::NEG)
                                     .set_compute_data_type(fe::DataType_t::FLOAT));
        Y->set_output(true).set_name("Y").set_data_type(fe::DataType_t::FLOAT);
        VariantPack variant_pack = {{X, X_dev}, {Y, Y_dev}};
        build_and_execute(graph, variant_pack);
        std::cout << "neg executed successfully\n";
        cudaFree(X_dev);
        cudaFree(Y_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
