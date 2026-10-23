// Complete standalone C++ sample.
// API page: docs/api/cpp/sdpa_fp8_backward.md
// API: sdpa_fp8_backward
// Use case: Basic
// Source: adapted from `samples/cpp/sdpa/fp8_bwd.cpp`.

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
        int64_t const B = 1, H = 2, S = 128, D = 128;
        int64_t const qkv_storage = B * S * 3 * H * D;
        int64_t const o_count     = B * S * H * D;
        int64_t const stat_count  = B * H * S;
        int8_t* QKV_dev           = managed<int8_t>(qkv_storage);
        int8_t* DQDKDV_dev        = managed<int8_t>(qkv_storage);
        int8_t* O_dev             = managed<int8_t>(o_count);
        int8_t* DO_dev            = managed<int8_t>(o_count);
        float* Stats_dev          = managed<float>(stat_count);
        float* Scalars_dev        = managed<float>(16);
        std::uniform_int_distribution<int> byte_dis(-64, 63);
        for (int64_t i = 0; i < qkv_storage; ++i) {
            QKV_dev[i]    = static_cast<int8_t>(byte_dis(gen));
            DQDKDV_dev[i] = static_cast<int8_t>(byte_dis(gen));
        }
        for (int64_t i = 0; i < o_count; ++i) {
            O_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            DO_dev[i] = static_cast<int8_t>(byte_dis(gen));
        }
        for (int64_t i = 0; i < (stat_count); ++i) Stats_dev[i] = dis(gen);
        for (int64_t i = 0; i < (16); ++i) Scalars_dev[i] = dis(gen);
        for (int64_t i = 0; i < 12; ++i) Scalars_dev[i] = std::fabs(Scalars_dev[i]) + 0.5f;

        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto dims         = std::vector<int64_t>{B, H, S, D};
        auto qkv_strides  = std::vector<int64_t>{S * 3 * H * D, D, 3 * H * D, 1};
        auto o_strides    = std::vector<int64_t>{S * H * D, D, H * D, 1};
        auto stat_dims    = std::vector<int64_t>{B, H, S, 1};
        auto stat_strides = std::vector<int64_t>{H * S, S, 1, 1};
        auto Q            = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto K            = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto V            = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto O =
            graph.tensor(fe::graph::Tensor_attributes().set_name("O").set_dim(dims).set_stride(o_strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto DO = graph.tensor(
            fe::graph::Tensor_attributes().set_name("dO").set_dim(dims).set_stride(o_strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto Stats     = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Stats")
                                      .set_dim(stat_dims)
                                      .set_stride(stat_strides)
                                      .set_data_type(fe::DataType_t::FLOAT));
        auto DescaleQ  = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("Descale_Q")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));
        auto DescaleK  = graph.tensor_like(DescaleQ, "Descale_K");
        auto DescaleV  = graph.tensor_like(DescaleQ, "Descale_V");
        auto DescaleO  = graph.tensor_like(DescaleQ, "Descale_O");
        auto DescaleDO = graph.tensor_like(DescaleQ, "Descale_dO");
        auto DescaleS  = graph.tensor_like(DescaleQ, "Descale_S");
        auto DescaleDP = graph.tensor_like(DescaleQ, "Descale_dP");
        auto ScaleS    = graph.tensor_like(DescaleQ, "Scale_S");
        auto ScaleDQ   = graph.tensor_like(DescaleQ, "Scale_dQ");
        auto ScaleDK   = graph.tensor_like(DescaleQ, "Scale_dK");
        auto ScaleDV   = graph.tensor_like(DescaleQ, "Scale_dV");
        auto ScaleDP   = graph.tensor_like(DescaleQ, "Scale_dP");
        auto attrs     = fe::graph::SDPA_fp8_backward_attributes()
                         .set_name("sdpa_fp8_backward")
                         .set_causal_mask(true)
                         .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)));
        auto [DQ, DK, DV, AmaxDQ, AmaxDK, AmaxDV, AmaxDP] = graph.sdpa_fp8_backward(Q,
                                                                                    K,
                                                                                    V,
                                                                                    O,
                                                                                    DO,
                                                                                    Stats,
                                                                                    DescaleQ,
                                                                                    DescaleK,
                                                                                    DescaleV,
                                                                                    DescaleO,
                                                                                    DescaleDO,
                                                                                    DescaleS,
                                                                                    DescaleDP,
                                                                                    ScaleS,
                                                                                    ScaleDQ,
                                                                                    ScaleDK,
                                                                                    ScaleDV,
                                                                                    ScaleDP,
                                                                                    attrs);
        DQ->set_output(true).set_dim(dims).set_stride(qkv_strides).set_data_type(fe::DataType_t::FP8_E4M3);
        DK->set_output(true).set_dim(dims).set_stride(qkv_strides).set_data_type(fe::DataType_t::FP8_E4M3);
        DV->set_output(true).set_dim(dims).set_stride(qkv_strides).set_data_type(fe::DataType_t::FP8_E4M3);
        AmaxDQ->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        AmaxDK->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        AmaxDV->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        AmaxDP->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        VariantPack variant_pack = {{Q, QKV_dev},
                                    {K, QKV_dev + H * D},
                                    {V, QKV_dev + 2 * H * D},
                                    {O, O_dev},
                                    {DO, DO_dev},
                                    {Stats, Stats_dev},
                                    {DQ, DQDKDV_dev},
                                    {DK, DQDKDV_dev + H * D},
                                    {DV, DQDKDV_dev + 2 * H * D},
                                    {DescaleQ, Scalars_dev},
                                    {DescaleK, Scalars_dev + 1},
                                    {DescaleV, Scalars_dev + 2},
                                    {DescaleO, Scalars_dev + 3},
                                    {DescaleDO, Scalars_dev + 4},
                                    {DescaleS, Scalars_dev + 5},
                                    {DescaleDP, Scalars_dev + 6},
                                    {ScaleS, Scalars_dev + 7},
                                    {ScaleDQ, Scalars_dev + 8},
                                    {ScaleDK, Scalars_dev + 9},
                                    {ScaleDV, Scalars_dev + 10},
                                    {ScaleDP, Scalars_dev + 11},
                                    {AmaxDQ, Scalars_dev + 12},
                                    {AmaxDK, Scalars_dev + 13},
                                    {AmaxDV, Scalars_dev + 14},
                                    {AmaxDP, Scalars_dev + 15}};
        build_and_execute(graph, variant_pack);
        std::cout << "sdpa_fp8_backward executed successfully\n";
        cudaFree(QKV_dev);
        cudaFree(DQDKDV_dev);
        cudaFree(O_dev);
        cudaFree(DO_dev);
        cudaFree(Stats_dev);
        cudaFree(Scalars_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
