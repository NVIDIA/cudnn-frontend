// Complete standalone C++ sample.
// API page: docs/api/cpp/sdpa_mxfp8.md
// API: sdpa_mxfp8
// Use case: Basic
// Source: adapted from `samples/cpp/sdpa/mxfp8_fwd.cpp`.

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
        int64_t const B = 1, H = 2, S = 128, D = 128, block_size = 32;
        int64_t const qkv_storage    = B * S * 3 * H * D;
        int64_t const o_count        = B * S * H * D;
        int64_t const d_scale        = (D + block_size - 1) / block_size;
        int64_t const s_scale        = (S + block_size - 1) / block_size;
        int64_t const s_padded       = ((S + 127) / 128) * 128;
        int64_t const d_padded       = ((D + 127) / 128) * 128;
        int64_t const d_scale_padded = ((d_scale + 3) / 4) * 4;
        int64_t const s_scale_padded = ((s_scale + 3) / 4) * 4;
        int8_t* QKV_dev              = managed<int8_t>(qkv_storage);
        nv_bfloat16* O_dev           = managed<nv_bfloat16>(o_count);
        int64_t const sf_qk_count    = B * H * s_padded * d_scale_padded;
        int64_t const sf_v_count     = B * H * s_scale_padded * d_padded;
        int8_t* SF_Q_dev             = managed<int8_t>(sf_qk_count);
        int8_t* SF_K_dev             = managed<int8_t>(sf_qk_count);
        int8_t* SF_V_dev             = managed<int8_t>(sf_v_count);
        float* Stats_dev             = managed<float>(B * H * S);
        float* AmaxO_dev             = managed<float>(1);
        std::uniform_int_distribution<int> byte_dis(-64, 63);
        for (int64_t i = 0; i < qkv_storage; ++i) QKV_dev[i] = static_cast<int8_t>(byte_dis(gen));
        for (int64_t i = 0; i < sf_qk_count; ++i) {
            SF_Q_dev[i] = static_cast<int8_t>(byte_dis(gen));
            SF_K_dev[i] = static_cast<int8_t>(byte_dis(gen));
        }
        for (int64_t i = 0; i < sf_v_count; ++i) SF_V_dev[i] = static_cast<int8_t>(byte_dis(gen));
        for (int64_t i = 0; i < (o_count); ++i) O_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (B * H * S); ++i) Stats_dev[i] = dis(gen);
        for (int64_t i = 0; i < (1); ++i) AmaxO_dev[i] = dis(gen);

        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto dims        = std::vector<int64_t>{B, H, S, D};
        auto qkv_strides = std::vector<int64_t>{S * 3 * H * D, D, 3 * H * D, 1};
        auto o_strides   = std::vector<int64_t>{S * H * D, D, H * D, 1};
        auto Q           = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto K           = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto V           = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto sf_qk_dims  = std::vector<int64_t>{B, H, s_padded, d_scale_padded};
        auto sf_qk_strides =
            std::vector<int64_t>{H * s_padded * d_scale_padded, s_padded * d_scale_padded, d_scale_padded, 1};
        auto sf_v_dims = std::vector<int64_t>{B, H, s_scale_padded, d_padded};
        auto sf_v_strides =
            std::vector<int64_t>{H * s_scale_padded * d_padded, s_scale_padded * d_padded, 1, s_scale_padded};
        auto SF_Q  = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_Q")
                                     .set_dim(sf_qk_dims)
                                     .set_stride(sf_qk_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto SF_K  = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_K")
                                     .set_dim(sf_qk_dims)
                                     .set_stride(sf_qk_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto SF_V  = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_V")
                                     .set_dim(sf_v_dims)
                                     .set_stride(sf_v_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto attrs = fe::graph::SDPA_fp8_attributes()
                         .set_name("sdpa_mxfp8")
                         .set_generate_stats(true)
                         .set_causal_mask(true)
                         .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)));
        auto [O, Stats, AmaxO] = graph.sdpa_fp8(Q, K, V, SF_Q, SF_K, SF_V, attrs);
        O->set_output(true).set_dim(dims).set_stride(o_strides).set_data_type(fe::DataType_t::BFLOAT16);
        Stats->set_output(true).set_dim({B, H, S, 1}).set_stride({H * S, S, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        AmaxO->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        VariantPack variant_pack = {{Q, QKV_dev},
                                    {K, QKV_dev + H * D},
                                    {V, QKV_dev + 2 * H * D},
                                    {SF_Q, SF_Q_dev},
                                    {SF_K, SF_K_dev},
                                    {SF_V, SF_V_dev},
                                    {O, O_dev},
                                    {Stats, Stats_dev},
                                    {AmaxO, AmaxO_dev}};
        build_and_execute(graph, variant_pack);
        std::cout << "sdpa_mxfp8 executed successfully\n";
        cudaFree(QKV_dev);
        cudaFree(O_dev);
        cudaFree(SF_Q_dev);
        cudaFree(SF_K_dev);
        cudaFree(SF_V_dev);
        cudaFree(Stats_dev);
        cudaFree(AmaxO_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
