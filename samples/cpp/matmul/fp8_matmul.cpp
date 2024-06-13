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

TEST_CASE("Matmul fp8 precision", "[matmul][graph]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    if ((is_hopper_arch() && cudnnGetVersion() >= 90000) == false) {
        SKIP("FP8 gemm not supported pre-Hopper or pre-cudnn-9.0.0");
    }

    namespace fe = cudnn_frontend;
    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Initialize input tensors with int8_t as proxy for fp8
    Surface<int8_t> A_gpu(b * m * k, false);
    Surface<int8_t> B_gpu(b * k * n, false);

    Surface<float> A_descale_gpu(1, false);
    Surface<float> B_descale_gpu(1, false);

    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::FP8_E4M3);
    auto A = graph.tensor(A_attributes);

    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, 1, k})
                            .set_data_type(fe::DataType_t::FP8_E4M3);
    auto B = graph.tensor(B_attributes);

    auto A_descale_attributes =
        fe::graph::Tensor_attributes().set_name("A").set_dim({1, 1, 1}).set_stride({1, 1, 1}).set_data_type(
            fe::DataType_t::FLOAT);
    auto B_descale_attributes =
        fe::graph::Tensor_attributes().set_name("B").set_dim({1, 1, 1}).set_stride({1, 1, 1}).set_data_type(
            fe::DataType_t::FLOAT);

    auto A_descale = graph.tensor(A_descale_attributes);
    auto B_descale = graph.tensor(B_descale_attributes);

    auto matmul_attributes =
        // fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_data_type(fe::DataType_t::FLOAT);

    // Add scale_A operation
    auto pw_0_attributes = fe::graph::Pointwise_attributes()
                               //    .set_name("pw0_Mul")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto C_after_pw_0 = graph.pointwise(C, A_descale, pw_0_attributes);
    C_after_pw_0->set_data_type(fe::DataType_t::FLOAT);

    // Add descale_B operation
    auto pw_1_attributes = fe::graph::Pointwise_attributes()
                               //    .set_name("pw1_Mul")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto C_after_pw_1 = graph.pointwise(C_after_pw_0, B_descale, pw_1_attributes);
    C_after_pw_1->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    std::cout << graph << std::endl;
    REQUIRE(graph.validate().is_good());

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<float> C_gpu(b * m * n, false);
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr},
        {B, B_gpu.devPtr},
        {C_after_pw_1, C_gpu.devPtr},
        {A_descale, A_descale_gpu.devPtr},
        {B_descale, B_descale_gpu.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}
