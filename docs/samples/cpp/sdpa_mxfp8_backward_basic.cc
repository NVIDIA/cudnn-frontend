// Complete standalone C++ sample.
// API page: docs/api/cpp/sdpa_mxfp8_backward.md
// API: sdpa_mxfp8_backward
// Use case: Basic
// Source: adapted from `samples/cpp/sdpa/mxfp8_bwd.cpp`.

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
        int64_t const q_count        = B * H * S * D;
        int64_t const stat_count     = B * H * S;
        int64_t const d_scale        = (D + block_size - 1) / block_size;
        int64_t const s_scale        = (S + block_size - 1) / block_size;
        int64_t const s_padded       = ((S + 127) / 128) * 128;
        int64_t const d_padded       = ((D + 127) / 128) * 128;
        int64_t const d_scale_padded = ((d_scale + 3) / 4) * 4;
        int64_t const s_scale_padded = ((s_scale + 3) / 4) * 4;
        int8_t* Q_dev                = managed<int8_t>(q_count);
        int8_t* QT_dev               = managed<int8_t>(q_count);
        int8_t* K_dev                = managed<int8_t>(q_count);
        int8_t* KT_dev               = managed<int8_t>(q_count);
        int8_t* V_dev                = managed<int8_t>(q_count);
        int8_t* DO_dev               = managed<int8_t>(q_count);
        int8_t* DOT_dev              = managed<int8_t>(q_count);
        nv_bfloat16* O_dev           = managed<nv_bfloat16>(q_count);
        nv_bfloat16* DOF16_dev       = managed<nv_bfloat16>(q_count);
        nv_bfloat16* DQ_dev          = managed<nv_bfloat16>(q_count);
        nv_bfloat16* DK_dev          = managed<nv_bfloat16>(q_count);
        nv_bfloat16* DV_dev          = managed<nv_bfloat16>(q_count);
        float* Stats_dev             = managed<float>(stat_count);
        int64_t const sf_qk_count    = B * H * s_padded * d_scale_padded;
        int64_t const sf_t_count     = B * H * s_scale_padded * d_padded;
        int8_t* SF_Q_dev             = managed<int8_t>(sf_qk_count);
        int8_t* SF_K_dev             = managed<int8_t>(sf_qk_count);
        int8_t* SF_V_dev             = managed<int8_t>(sf_qk_count);
        int8_t* SF_DO_dev            = managed<int8_t>(sf_qk_count);
        int8_t* SF_QT_dev            = managed<int8_t>(sf_t_count);
        int8_t* SF_KT_dev            = managed<int8_t>(sf_t_count);
        int8_t* SF_DOT_dev           = managed<int8_t>(sf_t_count);
        float* Amax_dev              = managed<float>(3);
        std::uniform_int_distribution<int> byte_dis(-64, 63);
        for (int64_t i = 0; i < q_count; ++i) {
            Q_dev[i]   = static_cast<int8_t>(byte_dis(gen));
            QT_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            K_dev[i]   = static_cast<int8_t>(byte_dis(gen));
            KT_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            V_dev[i]   = static_cast<int8_t>(byte_dis(gen));
            DO_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            DOT_dev[i] = static_cast<int8_t>(byte_dis(gen));
        }
        for (int64_t i = 0; i < sf_qk_count; ++i) {
            SF_Q_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            SF_K_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            SF_V_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            SF_DO_dev[i] = static_cast<int8_t>(byte_dis(gen));
        }
        for (int64_t i = 0; i < sf_t_count; ++i) {
            SF_QT_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            SF_KT_dev[i]  = static_cast<int8_t>(byte_dis(gen));
            SF_DOT_dev[i] = static_cast<int8_t>(byte_dis(gen));
        }
        for (int64_t i = 0; i < (q_count); ++i) O_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (q_count); ++i) DOF16_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (q_count); ++i) DQ_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (q_count); ++i) DK_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (q_count); ++i) DV_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (stat_count); ++i) Stats_dev[i] = dis(gen);
        for (int64_t i = 0; i < (3); ++i) Amax_dev[i] = dis(gen);

        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto dims         = std::vector<int64_t>{B, H, S, D};
        auto strides      = std::vector<int64_t>{S * H * D, D, H * D, 1};
        auto stat_dims    = std::vector<int64_t>{B, H, S, 1};
        auto stat_strides = std::vector<int64_t>{H * S, S, 1, 1};
        auto Q =
            graph.tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto QT =
            graph.tensor(fe::graph::Tensor_attributes().set_name("Q_T").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto K =
            graph.tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto KT =
            graph.tensor(fe::graph::Tensor_attributes().set_name("K_T").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto V =
            graph.tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto O = graph.tensor(
            fe::graph::Tensor_attributes().set_name("O_f16").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::BFLOAT16));
        auto DOF16 = graph.tensor(
            fe::graph::Tensor_attributes().set_name("dO_f16").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::BFLOAT16));
        auto DO =
            graph.tensor(fe::graph::Tensor_attributes().set_name("dO").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto DOT = graph.tensor(
            fe::graph::Tensor_attributes().set_name("dO_T").set_dim(dims).set_stride(strides).set_data_type(
                fe::DataType_t::FP8_E4M3));
        auto Stats     = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Stats")
                                      .set_dim(stat_dims)
                                      .set_stride(stat_strides)
                                      .set_data_type(fe::DataType_t::FLOAT));
        auto sf_d_dims = std::vector<int64_t>{B, H, s_padded, d_scale_padded};
        auto sf_d_strides =
            std::vector<int64_t>{H * s_padded * d_scale_padded, s_padded * d_scale_padded, d_scale_padded, 1};
        auto sf_t_dims = std::vector<int64_t>{B, H, s_scale_padded, d_padded};
        auto sf_t_strides =
            std::vector<int64_t>{H * s_scale_padded * d_padded, s_scale_padded * d_padded, s_scale_padded, 1};
        auto SF_Q   = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_Q")
                                     .set_dim(sf_d_dims)
                                     .set_stride(sf_d_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto SF_K   = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_K")
                                     .set_dim(sf_d_dims)
                                     .set_stride(sf_d_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto SF_V   = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_V")
                                     .set_dim(sf_d_dims)
                                     .set_stride(sf_d_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto SF_DO  = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("SF_dO")
                                      .set_dim(sf_d_dims)
                                      .set_stride(sf_d_strides)
                                      .set_data_type(fe::DataType_t::FP8_E8M0)
                                      .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto SF_QT  = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("SF_Q_T")
                                      .set_dim(sf_t_dims)
                                      .set_stride(sf_t_strides)
                                      .set_data_type(fe::DataType_t::FP8_E8M0)
                                      .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto SF_KT  = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("SF_K_T")
                                      .set_dim(sf_t_dims)
                                      .set_stride(sf_t_strides)
                                      .set_data_type(fe::DataType_t::FP8_E8M0)
                                      .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto SF_DOT = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("SF_dO_T")
                                       .set_dim(sf_t_dims)
                                       .set_stride(sf_t_strides)
                                       .set_data_type(fe::DataType_t::FP8_E8M0)
                                       .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        auto attrs  = fe::graph::SDPA_fp8_backward_attributes()
                         .set_name("sdpa_mxfp8_backward")
                         .set_causal_mask(true)
                         .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)))
                         .set_deterministic_algorithm(true);
        auto [DQ, DK, DV, AmaxDQ, AmaxDK, AmaxDV] = graph.sdpa_fp8_backward(
            Q, QT, K, KT, V, O, DOF16, DO, DOT, Stats, SF_Q, SF_QT, SF_K, SF_KT, SF_V, SF_DO, SF_DOT, attrs);
        DQ->set_output(true).set_dim(dims).set_stride(strides).set_data_type(fe::DataType_t::BFLOAT16);
        DK->set_output(true).set_dim(dims).set_stride(strides).set_data_type(fe::DataType_t::BFLOAT16);
        DV->set_output(true).set_dim(dims).set_stride(strides).set_data_type(fe::DataType_t::BFLOAT16);
        AmaxDQ->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        AmaxDK->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        AmaxDV->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
        VariantPack variant_pack = {{Q, Q_dev},
                                    {QT, QT_dev},
                                    {K, K_dev},
                                    {KT, KT_dev},
                                    {V, V_dev},
                                    {O, O_dev},
                                    {DOF16, DOF16_dev},
                                    {DO, DO_dev},
                                    {DOT, DOT_dev},
                                    {Stats, Stats_dev},
                                    {SF_Q, SF_Q_dev},
                                    {SF_QT, SF_QT_dev},
                                    {SF_K, SF_K_dev},
                                    {SF_KT, SF_KT_dev},
                                    {SF_V, SF_V_dev},
                                    {SF_DO, SF_DO_dev},
                                    {SF_DOT, SF_DOT_dev},
                                    {DQ, DQ_dev},
                                    {DK, DK_dev},
                                    {DV, DV_dev},
                                    {AmaxDQ, Amax_dev},
                                    {AmaxDK, Amax_dev + 1},
                                    {AmaxDV, Amax_dev + 2}};
        build_and_execute(graph, variant_pack);
        std::cout << "sdpa_mxfp8_backward executed successfully\n";
        cudaFree(Q_dev);
        cudaFree(QT_dev);
        cudaFree(K_dev);
        cudaFree(KT_dev);
        cudaFree(V_dev);
        cudaFree(DO_dev);
        cudaFree(DOT_dev);
        cudaFree(O_dev);
        cudaFree(DOF16_dev);
        cudaFree(DQ_dev);
        cudaFree(DK_dev);
        cudaFree(DV_dev);
        cudaFree(Stats_dev);
        cudaFree(SF_Q_dev);
        cudaFree(SF_K_dev);
        cudaFree(SF_V_dev);
        cudaFree(SF_DO_dev);
        cudaFree(SF_QT_dev);
        cudaFree(SF_KT_dev);
        cudaFree(SF_DOT_dev);
        cudaFree(Amax_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
