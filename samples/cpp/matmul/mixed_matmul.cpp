/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <catch2/catch_test_macros.hpp>

#include <random>

#include "../../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Mixed Precision Matmul", "[matmul][graph]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    namespace fe = cudnn_frontend;

    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Initialize input tensors
    Surface<int8_t> A_gpu(b * m * k, false);
    // note this is a bf16 tensor, but half is used just for memory allocation
    Surface<half> B_gpu(b * k * n, false);

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::INT8);
    auto A            = graph.tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, n, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto B = graph.tensor(B_attributes);

    // Cast the input tensors to required mma precision
    auto identity_attributes = fe::graph::Pointwise_attributes()
                                   .set_name("Cast_A")
                                   .set_mode(fe::PointwiseMode_t::IDENTITY)
                                   // INT8->FLOAT->BF16 to maintain precision
                                   .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A_casted = graph.pointwise(A, identity_attributes);
    A_casted->set_data_type(fe::DataType_t::BFLOAT16);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A_casted, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    REQUIRE(graph.validate().is_good());

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    if (is_hopper_arch() && cudnnGetVersion() >= 8906) {
        REQUIRE(graph.check_support(handle).is_good());
    } else {
        SKIP("int8_bf16 mixed precision gemm not supported pre-Hopper or pre-cudnn-8.9.6");
    }

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    //// Run cudnn graph
    // note this is a bf16 tensor, but half is used just for memory allocation
    Surface<half> C_gpu(b * m * n, false);
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};

    std::cout << graph.print() << std::endl;
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}
