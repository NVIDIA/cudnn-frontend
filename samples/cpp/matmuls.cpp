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

#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Matmul", "[matmul][graph]") {
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }
    namespace fe = cudnn_frontend;

    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Initialize input tensors
    Surface<half> A_gpu(b * m * k, false);
    Surface<half> B_gpu(b * k * n, false);

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto A            = graph.tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, n, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto B = graph.tensor(B_attributes);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph.validate().is_good());

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    // Run cudnn graph
    Surface<float> C_gpu(b * m * n, false);
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}

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
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_data_type(fe::DataType_t::FLOAT);

    // Add scale_A operation
    auto pw_0_attributes = fe::graph::Pointwise_attributes()
                               .set_name("pw0_Mul")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto C_after_pw_0 = graph.pointwise(C, A_descale, pw_0_attributes);
    C_after_pw_0->set_data_type(fe::DataType_t::FLOAT);

    // Add descale_B operation
    auto pw_1_attributes = fe::graph::Pointwise_attributes()
                               .set_name("pw1_Mul")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto C_after_pw_1 = graph.pointwise(C_after_pw_0, B_descale, pw_1_attributes);
    C_after_pw_1->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

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

    std::cout << graph.print() << std::endl;
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}

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

TEST_CASE("Int8 Matmul", "[matmul][graph]") {
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
    Surface<int8_t> B_gpu(b * k * n, false);

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
                            .set_stride({k * n, 1, n})
                            .set_data_type(fe::DataType_t::INT8);
    auto B = graph.tensor(B_attributes);

    auto Bias_attributes = cudnn_frontend::graph::Tensor_attributes()
                               .set_name("Bias")
                               .set_dim({b, m, n})
                               .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                               .set_stride({m * n, n, 1});
    auto Bias = graph.tensor(Bias_attributes);

    // Add MATMUL operation
    auto matmul_attributes = cudnn_frontend::graph::Matmul_attributes()
                                 .set_compute_data_type(cudnn_frontend::DataType_t::INT32)
                                 .set_name("GEMM");
    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_data_type(cudnn_frontend::DataType_t::FLOAT);

    // Add ADD operation
    auto add_attributes = cudnn_frontend::graph::Pointwise_attributes()
                              .set_name("pw1_add")
                              .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                              .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    auto C_after_add = graph.pointwise(C, Bias, add_attributes);
    C_after_add->set_output(true).set_data_type(cudnn_frontend::DataType_t::FLOAT);
    REQUIRE(graph.validate().is_good());

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    if (check_device_arch_newer_than("ampere") && cudnnGetVersion() >= 8906) {
        REQUIRE(graph.check_support(handle).is_good());
    } else {
        SKIP("int8 gemm not supported pre-Ampere or pre-cudnn-8.9.6");
    }

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    // Run cudnn graph
    // note this is a bf16 tensor, but half is used just for memory allocation
    Surface<float> C_gpu(b * m * n, false);
    Surface<float> Bias_gpu(b * m * n, false);
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C_after_add, C_gpu.devPtr}, {Bias, Bias_gpu.devPtr}};

    std::cout << graph.print() << std::endl;
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}

TEST_CASE("Abs + Matmul", "[matmul][graph]") {
    namespace fe = cudnn_frontend;

    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Initialize input tensors
    Surface<half> A_gpu(b * m * k, false);
    Surface<half> B_gpu(b * k * n, false);

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto A            = graph.tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, n, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto B = graph.tensor(B_attributes);

    // Add abs operation
    auto pw_0_attributes = fe::graph::Pointwise_attributes()
                               .set_name("pw0_Abs")
                               .set_mode(fe::PointwiseMode_t::ABS)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A_after_pw_0 = graph.pointwise(A, pw_0_attributes);
    A_after_pw_0->set_data_type(fe::DataType_t::BFLOAT16);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A_after_pw_0, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph.validate().is_good());

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    // Run cudnn graph
    Surface<float> C_gpu(b * m * n, false);
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}

TEST_CASE("Bias + Matmul", "[matmul][graph]") {
    namespace fe = cudnn_frontend;

    if (cudnnGetVersion() < 8600) {
        SKIP("Test requires cuDNN version 8.6.0 or above");
        return;
    }

    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Initialize input tensors
    Surface<half> A_gpu(b * m * k, false);
    Surface<half> B_gpu(b * k * n, false);
    Surface<half> Bias_gpu(b * m * 1, false);

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto A            = graph.tensor(A_attributes);
    auto B_attributes = fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({b, k, n})
                            .set_stride({k * n, n, 1})
                            .set_data_type(fe::DataType_t::BFLOAT16);
    auto B = graph.tensor(B_attributes);

    // Create Bias vector
    auto Bias_attributes =
        fe::graph::Tensor_attributes().set_name("Bias").set_dim({b, m, 1}).set_stride({m, 1, 1}).set_data_type(
            fe::DataType_t::BFLOAT16);
    auto Bias = graph.tensor(Bias_attributes);

    // Add ADD operation
    auto pw_0_attributes = fe::graph::Pointwise_attributes()
                               .set_name("pw0_Add")
                               .set_mode(fe::PointwiseMode_t::ADD)
                               .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A_after_pw_0 = graph.pointwise(A, Bias, pw_0_attributes);
    A_after_pw_0->set_data_type(fe::DataType_t::BFLOAT16);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A_after_pw_0, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph.validate().is_good());

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    int64_t plan_count = graph.get_execution_plan_count();

    std::vector<int64_t> successful_plans;
    std::vector<int64_t> unsuccessful_plans;
    for (int64_t plan_index = 0; plan_index < plan_count; plan_index++) {
        bool did_build_successfully = graph.build_plan_at_index(handle, plan_index).is_good();
        if (did_build_successfully) {
            successful_plans.push_back(plan_index);
        } else {
            unsuccessful_plans.push_back(plan_index);
        }
    }

    // Run cudnn graph
    Surface<float> C_gpu(b * m * n, false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}, {Bias, Bias_gpu.devPtr}};

    // Run a unsuccessful engine and except error
    std::vector<int64_t> random_unsuccessful;
    std::sample(unsuccessful_plans.begin(),
                unsuccessful_plans.end(),
                std::back_inserter(random_unsuccessful),
                1,
                std::mt19937{std::random_device{}()});
    if (random_unsuccessful.size()) {
        REQUIRE(graph.execute_plan_at_index(handle, variant_pack, nullptr, random_unsuccessful.front()).is_bad());
    }

    // Run a successful engine and except success
    std::vector<int64_t> random_successful;
    std::sample(successful_plans.begin(),
                successful_plans.end(),
                std::back_inserter(random_successful),
                1,
                std::mt19937{std::random_device{}()});
    Surface<int8_t> workspace(graph.get_workspace_size_plan_at_index(random_successful.front()), false);
    REQUIRE(graph.execute_plan_at_index(handle, variant_pack, workspace.devPtr, random_successful.front()).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}

TEST_CASE("Matmul SBR Graph", "[matmul][graph]") {
    namespace fe = cudnn_frontend;

    auto b = 4;
    auto m = 16;
    auto k = 64;
    auto n = 32;

    using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // A
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // B
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // bias
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // S
                                         std::shared_ptr<fe::graph::Tensor_attributes>   // O
                                         >;

    std::unordered_map<std::size_t, graph_and_tensors> user_maintained_cache;

    auto lookup_cache_or_build_graph =
        [b, m, n, k, &user_maintained_cache](
            cudnnHandle_t handle, void* A_ptr, void* B_ptr, void* scale_ptr, void* bias_ptr, void* O_ptr) {
            auto graph = std::make_shared<fe::graph::Graph>();
            graph->set_io_data_type(fe::DataType_t::HALF)
                .set_intermediate_data_type(fe::DataType_t::FLOAT)
                .set_compute_data_type(fe::DataType_t::FLOAT);

            auto A = graph->tensor(
                fe::graph::Tensor_attributes().set_name("A").set_dim({b, m, k}).set_stride({m * k, 1, m}));

            auto B = graph->tensor(
                fe::graph::Tensor_attributes().set_name("B").set_dim({b, k, n}).set_stride({n * k, 1, k}));

            fe::graph::Matmul_attributes matmul;
            auto C = graph->matmul(A, B, matmul);

            auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
            auto S             = graph->tensor(
                fe::graph::Tensor_attributes().set_name("scale").set_dim({b, m, n}).set_stride({m * n, n, 1}));
            auto scale_output = graph->pointwise(C, S, scale_options);

            auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
            auto bias         = graph->tensor_like(S);
            bias->set_name("bias");
            auto bias_output = graph->pointwise(scale_output, bias, bias_options);

            auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
            auto O            = graph->pointwise(bias_output, relu_options);
            O->set_output(true);

            REQUIRE(graph->validate().is_good());

            auto key = graph->key();

            auto it = user_maintained_cache.find(key);

            if (it != user_maintained_cache.end()) {
                return it->second;
            }

            REQUIRE(graph->build_operation_graph(handle).is_good());

            REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

            REQUIRE(graph->check_support(handle).is_good());

            REQUIRE(graph->build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good());

            Surface<int8_t> autotune_workspace(graph->get_autotune_workspace_size(), false);

            std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                {A, A_ptr}, {B, B_ptr}, {S, scale_ptr}, {bias, bias_ptr}, {O, O_ptr}};

            REQUIRE(graph->autotune(handle, variant_pack, autotune_workspace.devPtr).is_good());

            (void)variant_pack;
            user_maintained_cache.insert({key, std::make_tuple(graph, A, B, bias, S, O)});

            return std::make_tuple(graph, A, B, bias, S, O);
        };

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    Surface<half> x_tensor(4 * 16 * 64, false);
    Surface<half> w_tensor(4 * 64 * 32, false);
    Surface<half> s_tensor(4 * 16 * 32, false);
    Surface<half> b_tensor(4 * 16 * 32, false);
    Surface<half> y_tensor(4 * 16 * 32, false);

    auto [graph, A, B, bias, scale, O] = lookup_cache_or_build_graph(
        handle, x_tensor.devPtr, w_tensor.devPtr, s_tensor.devPtr, b_tensor.devPtr, y_tensor.devPtr);

    Surface<int8_t> workspace(graph->get_workspace_size(), false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{A, x_tensor.devPtr},
                                                                                             {B, w_tensor.devPtr},
                                                                                             {scale, s_tensor.devPtr},
                                                                                             {bias, b_tensor.devPtr},
                                                                                             {O, y_tensor.devPtr}};
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}
