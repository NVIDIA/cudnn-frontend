// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/moe_grouped_matmul_bwd.md
// API: moe_grouped_matmul_bwd
// Use case: Training
// Source: adapted from `samples/cpp/moe_grouped_matmul/moe_grouped_matmul.cpp`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

void
check(fe::error_t status) {
    if (!status.is_good()) throw std::runtime_error(status.get_message());
}

template <typename T>
T*
managed(int64_t count) {
    T* pointer = nullptr;
    cudaMallocManaged(&pointer, count * sizeof(T));
    return pointer;
}

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;

void
build_and_execute(fe::graph::Graph& graph, VariantPack& variant_pack) {
    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);
    check(graph.build(handle, {fe::HeurMode_t::A}));

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
#if CUDNN_VERSION < 92200
    std::cout << "SKIP: moe_grouped_matmul_bwd requires cuDNN 9.22 or newer\n";
    return 0;
#else
    int64_t const experts              = 36;
    int64_t const rows                 = 2000;
    int64_t const hidden               = 520;
    int64_t const output               = 248;
    std::vector<int32_t> const offsets = {0,    1,    2,    3,    4,    5,    6,    7,    8,
                                          9,    10,   11,   12,   13,   14,   15,   16,   17,
                                          18,   127,  255,  383,  483,  515,  643,  718,  924,
                                          1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900};

    nv_bfloat16* doutput_data = managed<nv_bfloat16>(rows * output);
    nv_bfloat16* token_data   = managed<nv_bfloat16>(rows * hidden);
    int32_t* offset_data      = managed<int32_t>(experts);
    nv_bfloat16* dweight_data = managed<nv_bfloat16>(experts * hidden * output);

    std::mt19937 generator(7);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int64_t index = 0; index < rows * output; ++index) {
        doutput_data[index] = __float2bfloat16(distribution(generator));
    }
    for (int64_t index = 0; index < rows * hidden; ++index) {
        token_data[index] = __float2bfloat16(distribution(generator));
    }
    std::copy(offsets.begin(), offsets.end(), offset_data);

    fe::graph::Graph graph{};
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto doutput_tensor = graph.tensor(fe::graph::Tensor_attributes()
                                           .set_name("doutput")
                                           .set_dim({1, rows, output})
                                           .set_stride({rows * output, output, 1})
                                           .set_data_type(fe::DataType_t::BFLOAT16));
    auto token_tensor   = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("token")
                                         .set_dim({1, rows, hidden})
                                         .set_stride({rows * hidden, hidden, 1})
                                         .set_data_type(fe::DataType_t::BFLOAT16));
    auto offsets_tensor = graph.tensor(fe::graph::Tensor_attributes()
                                           .set_name("first_token_offset")
                                           .set_dim({experts, 1, 1})
                                           .set_stride({1, 1, 1})
                                           .set_data_type(fe::DataType_t::INT32));
    auto attributes     = fe::graph::Moe_grouped_matmul_bwd_attributes()
                          .set_name("moe_grouped_matmul_bwd")
                          .set_compute_data_type(fe::DataType_t::FLOAT);
    auto dweight_tensor =
        graph.moe_grouped_matmul_bwd(doutput_tensor, token_tensor, offsets_tensor, attributes);
    dweight_tensor->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    VariantPack variant_pack = {{doutput_tensor, doutput_data},
                                {token_tensor, token_data},
                                {offsets_tensor, offset_data},
                                {dweight_tensor, dweight_data}};
    build_and_execute(graph, variant_pack);

    float const tolerance = 2.0e-2f * std::max(1.0f, std::sqrt(static_cast<float>(rows) / 128.0f));
    for (int64_t expert = 0; expert < experts; ++expert) {
        int64_t const start = offsets[expert];
        int64_t const end   = expert + 1 < experts ? offsets[expert + 1] : rows;
        for (int64_t n = 0; n < output; ++n) {
            for (int64_t h = 0; h < hidden; ++h) {
                float expected = 0.0f;
                for (int64_t row = start; row < end; ++row) {
                    expected += __bfloat162float(token_data[row * hidden + h]) *
                                __bfloat162float(doutput_data[row * output + n]);
                }
                int64_t const index = expert * hidden * output + n * hidden + h;
                float const actual  = __bfloat162float(dweight_data[index]);
                if (!std::isfinite(actual) ||
                    std::fabs(actual - expected) > tolerance + tolerance * std::fabs(expected)) {
                    throw std::runtime_error(
                        "dweight does not match the per-expert dense reference");
                }
            }
        }
    }
    std::cout << "moe grouped matmul backward ok: dweight verified per expert\n";

    cudaFree(doutput_data);
    cudaFree(token_data);
    cudaFree(offset_data);
    cudaFree(dweight_data);
    return 0;
#endif
}
