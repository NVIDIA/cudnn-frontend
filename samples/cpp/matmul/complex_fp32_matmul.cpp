/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <random>

#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Complex FP32 Matmul", "[matmul][graph]") {
    if (cudnnGetCudartVersion() < 12000 || cudnnGetVersion() < 91400) {
        SKIP("Test requires cuda toolkit 12.0 or above and cudnn version 9.14.0 or above");
    }
    namespace fe = cudnn_frontend;

    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Initialize input tensors
    // COMPLEX_FP32 datatype is a {float, float} in memory
    Surface<int8_t> A_gpu(2 * sizeof(float) * b * m * k);

    Surface<int8_t> B_gpu(2 * sizeof(float) * b * k * n);

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.

    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::COMPLEX_FP32);
    auto A = graph.tensor(A_attributes);

    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, 1, k})
                            .set_data_type(fe::DataType_t::COMPLEX_FP32);
    auto B = graph.tensor(B_attributes);

    // Add MATMUL operation
    auto matmul_attributes = cudnn_frontend::graph::Matmul_attributes()
                                 .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT)
                                 .set_name("GEMM");

    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_output(true).set_data_type(cudnn_frontend::DataType_t::COMPLEX_FP32);

    REQUIRE(graph.validate().is_good());

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    if (cudnnGetVersion() >= 9140) {
        REQUIRE(graph.check_support().is_good());
    } else {
        SKIP("complex gemm not supported pre-cudnn-9.14.0");
    }

    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    // Run cudnn graph
    Surface<float> C_gpu(2 * sizeof(float) * b * m * n);

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};

    std::cout << graph.print() << std::endl;
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}