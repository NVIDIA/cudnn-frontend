// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/moe_grouped_matmul.md
// API: moe_grouped_matmul
// Use case: Basic
// Source: adapted from `samples/cpp/moe_grouped_matmul/moe_grouped_matmul.cpp`.

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
    int64_t const num_experts = 2, token_num = 64, hidden_size = 128, weight_size = 64;
    int64_t const token_count     = token_num * hidden_size;
    int64_t const weight_count    = num_experts * hidden_size * weight_size;
    int64_t const output_count    = token_num * weight_size;
    nv_bfloat16* Token_dev        = managed<nv_bfloat16>(token_count);
    nv_bfloat16* Weight_dev       = managed<nv_bfloat16>(weight_count);
    nv_bfloat16* Output_dev       = managed<nv_bfloat16>(output_count);
    int32_t* FirstTokenOffset_dev = managed<int32_t>(num_experts);
    for (int64_t i = 0; i < (token_count); ++i) Token_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t i = 0; i < (weight_count); ++i) Weight_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t i = 0; i < (output_count); ++i) Output_dev[i] = __float2bfloat16(dis(gen));
    FirstTokenOffset_dev[0] = 0;
    FirstTokenOffset_dev[1] = static_cast<int32_t>(token_num / 2);

    fe::graph::Graph graph{};
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto Token            = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("token")
                                  .set_dim({1, token_num, hidden_size})
                                  .set_stride({token_num * hidden_size, hidden_size, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
    auto Weight           = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("weight")
                                   .set_dim({num_experts, hidden_size, weight_size})
                                   .set_stride({hidden_size * weight_size, 1, hidden_size})
                                   .set_data_type(fe::DataType_t::BFLOAT16));
    auto FirstTokenOffset = graph.tensor(fe::graph::Tensor_attributes()
                                             .set_name("first_token_offset")
                                             .set_dim({num_experts, 1, 1})
                                             .set_stride({1, 1, 1})
                                             .set_data_type(fe::DataType_t::INT32));
    auto attrs            = fe::graph::Moe_grouped_matmul_attributes()
                     .set_name("moe_grouped_matmul")
                     .set_mode(fe::MoeGroupedMatmulMode_t::NONE)
                     .set_top_k(1)
                     .set_compute_data_type(fe::DataType_t::FLOAT);
    auto Output =
        graph.moe_grouped_matmul(Token, Weight, FirstTokenOffset, nullptr, nullptr, attrs);
    Output->set_output(true)
        .set_dim({1, token_num, weight_size})
        .set_stride({token_num * weight_size, weight_size, 1})
        .set_data_type(fe::DataType_t::BFLOAT16);
    VariantPack variant_pack = {{Token, Token_dev},
                                {Weight, Weight_dev},
                                {FirstTokenOffset, FirstTokenOffset_dev},
                                {Output, Output_dev}};
    build_and_execute(graph, variant_pack);
    for (int64_t token = 0; token < token_num; ++token) {
        auto expert = token < FirstTokenOffset_dev[1] ? 0 : 1;
        for (int64_t n = 0; n < weight_size; ++n) {
            float expected = 0.0f;
            for (int64_t k = 0; k < hidden_size; ++k) {
                auto x = __bfloat162float(Token_dev[token * hidden_size + k]);
                auto w = __bfloat162float(
                    Weight_dev[expert * hidden_size * weight_size + k + n * hidden_size]);
                expected += x * w;
            }
            auto actual = __bfloat162float(Output_dev[token * weight_size + n]);
            if (!std::isfinite(actual) || !std::isfinite(expected) ||
                std::fabs(actual - expected) > 5.0e-1f) {
                throw std::runtime_error("moe grouped matmul result mismatch");
            }
        }
    }
    std::cout << "moe grouped matmul result matches reference\n";
    cudaFree(Token_dev);
    cudaFree(Weight_dev);
    cudaFree(Output_dev);
    cudaFree(FirstTokenOffset_dev);
    return 0;
}
