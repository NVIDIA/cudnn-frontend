/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Boolean CMP_GT and LOGICAL_AND fusion", "[membound][boolean][pointwise][graph]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 13010
    SKIP("Test requires cuda toolkit 13.1 or above");
    return;
#endif

#if (CUDNN_VERSION < 92200)
    SKIP("Boolean fusion sample requires cuDNN 9.22.0 or newer.");
#endif
    if (cudnn_frontend::detail::get_backend_version() < 92200) {
        SKIP("Boolean fusion sample requires cuDNN backend 9.22.0 or newer at runtime.");
    }
    if (!is_blackwell_computing_arch()) {
        SKIP("Boolean fusion (TensorIR MemBound engine) is only supported on Blackwell (data center Blackwell)");
    }

    constexpr int64_t d0 = 4, d1 = 8, d2 = 16;
    constexpr int64_t numel = d0 * d1 * d2;

    // Row-major strides for a 3D tensor
    constexpr int64_t s0 = d1 * d2;
    constexpr int64_t s1 = d2;
    constexpr int64_t s2 = 1;

    auto boolean_storage_type = fe::DataType_t::BOOLEAN;
#if (CUDNN_VERSION >= 92500)
    if (cudnn_frontend::detail::get_backend_version() >= 92500) {
        boolean_storage_type = fe::DataType_t::BYTE_BOOLEAN;
    }
#endif

    fe::graph::Graph graph{};
    graph.set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({d0, d1, d2})
                              .set_stride({s0, s1, s2})
                              .set_data_type(fe::DataType_t::HALF));

    auto threshold = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("threshold")
                                      .set_dim({d0, d1, d2})
                                      .set_stride({s0, s1, s2})
                                      .set_data_type(fe::DataType_t::FLOAT));

    auto B = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({d0, d1, d2})
                              .set_stride({s0, s1, s2})
                              .set_data_type(boolean_storage_type));

    auto after_cmp = graph.pointwise(X,
                                     threshold,
                                     fe::graph::Pointwise_attributes()
                                         .set_name("cmp_gt")
                                         .set_mode(fe::PointwiseMode_t::CMP_GT)
                                         .set_compute_data_type(fe::DataType_t::FLOAT));
    after_cmp->set_data_type(boolean_storage_type);

    auto Y = graph.pointwise(after_cmp,
                             B,
                             fe::graph::Pointwise_attributes()
                                 .set_name("logical_and")
                                 .set_mode(fe::PointwiseMode_t::LOGICAL_AND)
                                 .set_compute_data_type(fe::DataType_t::BOOLEAN));
    Y->set_output(true).set_data_type(boolean_storage_type);

    REQUIRE(graph.validate().is_good());

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<half> X_gpu(numel);
    Surface<float> threshold_gpu(numel);
    Surface<uint8_t> B_gpu(numel);
    Surface<uint8_t> Y_gpu(numel, 0);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_gpu.devPtr}, {threshold, threshold_gpu.devPtr}, {B, B_gpu.devPtr}, {Y, Y_gpu.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    // Verify against CPU reference
    std::vector<half> x_host(numel);
    std::vector<float> thresh_host(numel);
    std::vector<uint8_t> b_host(numel);
    std::vector<uint8_t> y_host(numel);

    CUDA_CHECK(cudaMemcpy(x_host.data(), X_gpu.devPtr, numel * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(thresh_host.data(), threshold_gpu.devPtr, numel * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_host.data(), B_gpu.devPtr, numel * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_host.data(), Y_gpu.devPtr, numel * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    int mismatches = 0;
    for (int64_t i = 0; i < numel; i++) {
        uint8_t expected = (__half2float(x_host[i]) > thresh_host[i]) && b_host[i] ? 1 : 0;
        if (y_host[i] != expected) {
            mismatches++;
        }
    }
    REQUIRE(mismatches == 0);
}
