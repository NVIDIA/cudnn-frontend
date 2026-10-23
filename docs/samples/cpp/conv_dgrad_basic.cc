// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/conv_dgrad.md
// API: conv_dgrad
// Use case: Basic
// Source: adapted from `samples/cpp/convolution/dgrads.cpp`.

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
    float* dY_dev = managed<float>(1 * 4 * 8 * 8);
    float* W_dev  = managed<float>(4 * 3 * 3 * 3);
    float* Y_dev  = managed<float>(1 * 3 * 8 * 8);
    for (int64_t i = 0; i < (1 * 4 * 8 * 8); ++i) dY_dev[i] = dis(gen);
    for (int64_t i = 0; i < (4 * 3 * 3 * 3); ++i) W_dev[i] = dis(gen);
    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X  = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({1, 3, 8, 8})
                              .set_stride({192, 64, 8, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto W  = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("W")
                              .set_dim({4, 3, 3, 3})
                              .set_stride({27, 9, 3, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto dY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("dY")
                               .set_dim({1, 4, 8, 8})
                               .set_stride({256, 64, 8, 1})
                               .set_data_type(fe::DataType_t::FLOAT));
    auto Y  = graph.conv_dgrad(dY,
                              W,
                              fe::graph::Conv_dgrad_attributes()
                                  .set_name("conv_dgrad")
                                  .set_padding({1, 1})
                                  .set_stride({1, 1})
                                  .set_dilation({1, 1})
                                  .set_compute_data_type(fe::DataType_t::FLOAT));
    Y->set_output(true)
        .set_name("Y")
        .set_dim({1, 3, 8, 8})
        .set_stride({192, 64, 8, 1})
        .set_data_type(fe::DataType_t::FLOAT);
    VariantPack variant_pack = {{dY, dY_dev}, {W, W_dev}, {Y, Y_dev}};
    build_and_execute(graph, variant_pack);
    for (int64_t c = 0; c < 3; ++c) {
        for (int64_t ih = 0; ih < 8; ++ih) {
            for (int64_t iw = 0; iw < 8; ++iw) {
                float expected = 0.0f;
                for (int64_t k = 0; k < 4; ++k) {
                    for (int64_t r = 0; r < 3; ++r) {
                        for (int64_t s = 0; s < 3; ++s) {
                            auto oh = ih + 1 - r;
                            auto ow = iw + 1 - s;
                            if (oh >= 0 && oh < 8 && ow >= 0 && ow < 8) {
                                expected += dY_dev[(k * 8 + oh) * 8 + ow] *
                                            W_dev[((k * 3 + c) * 3 + r) * 3 + s];
                            }
                        }
                    }
                }
                auto actual = Y_dev[(c * 8 + ih) * 8 + iw];
                if (!std::isfinite(actual) || !std::isfinite(expected) ||
                    std::fabs(actual - expected) > 1.0e-3f) {
                    throw std::runtime_error("conv_dgrad result mismatch");
                }
            }
        }
    }
    std::cout << "conv_dgrad result matches reference\n";
    cudaFree(dY_dev);
    cudaFree(W_dev);
    cudaFree(Y_dev);
    return 0;
}
