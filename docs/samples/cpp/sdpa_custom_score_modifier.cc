// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/sdpa.md
// API: sdpa
// Use case: Custom score modifier function
// Source: adapted from the `Softcap` helper in
// `include/cudnn_frontend/utils/attn_score_modifiers.h`.

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
                throw std::runtime_error(std::string("cudaMalloc(workspace): ") +
                                         cudaGetErrorString(cuda_status));
            }
        }

        status = graph.execute(handle, variant_pack, workspace);
        if (!status.is_good()) {
            throw std::runtime_error(std::string("graph.execute: ") + status.get_message());
        }

        cudaError_t cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaDeviceSynchronize: ") +
                                     cudaGetErrorString(cuda_status));
        }
    } catch (...) {
        if (workspace) cudaFree(workspace);
        cudnnDestroy(handle);
        throw;
    }

    if (workspace) {
        cudaError_t cuda_status = cudaFree(workspace);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaFree(workspace): ") +
                                     cudaGetErrorString(cuda_status));
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
        int64_t const B = 1, Hq = 2, Hkv = 1, S = 16, D = 64;
        nv_bfloat16* Q_dev = managed<nv_bfloat16>(B * Hq * S * D);
        nv_bfloat16* K_dev = managed<nv_bfloat16>(B * Hkv * S * D);
        nv_bfloat16* V_dev = managed<nv_bfloat16>(B * Hkv * S * D);
        nv_bfloat16* O_dev = managed<nv_bfloat16>(B * Hq * S * D);
        float* Stats_dev   = managed<float>(B * Hq * S);
        for (int64_t i = 0; i < (B * Hq * S * D); ++i) Q_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (B * Hkv * S * D); ++i) K_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (B * Hkv * S * D); ++i) V_dev[i] = __float2bfloat16(dis(gen));
        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto Q       = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim({B, Hq, S, D})
                                  .set_stride({Hq * S * D, D, Hq * D, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto K       = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({B, Hkv, S, D})
                                  .set_stride({Hkv * S * D, D, Hkv * D, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto V       = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({B, Hkv, S, D})
                                  .set_stride({Hkv * S * D, D, Hkv * D, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto Softcap = graph.tensor(50.0f);
        auto attrs   = fe::graph::SDPA_attributes()
                         .set_name("sdpa")
                         .set_generate_stats(false)
                         .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)))
                         .set_compute_data_type(fe::DataType_t::FLOAT);
        attrs.set_score_mod([Softcap](std::shared_ptr<fe::graph::Graph> g,
                                      std::shared_ptr<fe::graph::Tensor_attributes> scores) {
            auto div = g->pointwise(scores,
                                    Softcap,
                                    fe::graph::Pointwise_attributes()
                                        .set_name("softcap_div")
                                        .set_mode(fe::PointwiseMode_t::DIV)
                                        .set_compute_data_type(fe::DataType_t::FLOAT));
            auto t   = g->pointwise(div,
                                  fe::graph::Pointwise_attributes()
                                      .set_name("softcap_tanh")
                                      .set_mode(fe::PointwiseMode_t::TANH_FWD)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));
            return g->pointwise(t,
                                Softcap,
                                fe::graph::Pointwise_attributes()
                                    .set_name("softcap_mul")
                                    .set_mode(fe::PointwiseMode_t::MUL)
                                    .set_compute_data_type(fe::DataType_t::FLOAT));
        });
        auto outputs = graph.sdpa(Q, K, V, attrs);
        outputs[0]
            ->set_output(true)
            .set_name("O")
            .set_dim({B, Hq, S, D})
            .set_stride({Hq * S * D, D, Hq * D, 1})
            .set_data_type(fe::DataType_t::BFLOAT16);
        if (outputs[1] != nullptr)
            outputs[1]
                ->set_output(true)
                .set_name("Stats")
                .set_dim({B, Hq, S, 1})
                .set_stride({Hq * S, S, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT);
        VariantPack variant_pack = {{Q, Q_dev}, {K, K_dev}, {V, V_dev}, {outputs[0], O_dev}};
        if (outputs[1] != nullptr) variant_pack[outputs[1]] = Stats_dev;
        build_and_execute(graph, variant_pack);
        std::cout << "sdpa executed successfully\n";
        cudaFree(Q_dev);
        cudaFree(K_dev);
        cudaFree(V_dev);
        cudaFree(O_dev);
        cudaFree(Stats_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
