/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

void
matmul_dynamic_shapes(bool use_abs = false, bool use_bias = false) {
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }
    namespace fe = cudnn_frontend;

    // clang-format off
    struct {
        int64_t b,    m,    n,    k;
    } matmul_shapes[] = {
        {      16,   32,   32,  128},
        {      16,   64,   64,  128},
        {      16,   80,   80,  128},
        {      32,  128,  128,  256},
        {      32,   64,   64,  256},
    };
    // clang-format on

    constexpr int matmul_shapes_count = sizeof(matmul_shapes) / sizeof(matmul_shapes[0]);
    int64_t max_a_volume = 0, max_b_volume = 0, max_c_volume = 0, max_bias_volume = 0;
    for (int idx_shape = 0; idx_shape < matmul_shapes_count; ++idx_shape) {
        const auto& matmul_shape = matmul_shapes[idx_shape];
        max_a_volume             = std::max(max_a_volume, matmul_shape.b * matmul_shape.m * matmul_shape.k);
        max_b_volume             = std::max(max_b_volume, matmul_shape.b * matmul_shape.k * matmul_shape.n);
        max_c_volume             = std::max(max_c_volume, matmul_shape.b * matmul_shape.m * matmul_shape.n);
        max_bias_volume          = std::max(max_bias_volume, matmul_shape.b * matmul_shape.m);
    }

    auto kernel_cache = std::make_shared<fe::KernelCache>();

    const auto build_new_graph = [&matmul_shapes, &kernel_cache, &use_abs, &use_bias](cudnnHandle_t handle,
                                                                                      int idx_shape) {
        const auto& matmul_shape = matmul_shapes[idx_shape];

        // Make cudnn graph
        fe::graph::Graph graph{};

        graph.set_dynamic_shape_enabled(true).set_kernel_cache(kernel_cache);

        // Create the two non-virtual input tensors A and B.
        // There are read from global memory.
        auto A_attributes = fe::graph::Tensor_attributes()
                                .set_name("A")
                                .set_dim({matmul_shape.b, matmul_shape.m, matmul_shape.k})
                                .set_stride({matmul_shape.m * matmul_shape.k, matmul_shape.k, 1})
                                .set_data_type(fe::DataType_t::BFLOAT16);
        auto A = graph.tensor(A_attributes);

        auto B_attributes = fe::graph::Tensor_attributes()
                                .set_name("B")
                                .set_dim({matmul_shape.b, matmul_shape.k, matmul_shape.n})
                                .set_stride({matmul_shape.k * matmul_shape.n, matmul_shape.n, 1})
                                .set_data_type(fe::DataType_t::BFLOAT16);
        auto B = graph.tensor(B_attributes);

        auto matmul_attributes = fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT);

        std::shared_ptr<fe::graph::Tensor_attributes> C;
        std::shared_ptr<fe::graph::Tensor_attributes> Bias;

        if (use_abs) {
            // Add abs operation
            auto pw_0_attributes = fe::graph::Pointwise_attributes()
                                       .set_name("pw0_Abs")
                                       .set_mode(fe::PointwiseMode_t::ABS)
                                       .set_compute_data_type(fe::DataType_t::FLOAT);

            auto A_after_pw_0 = graph.pointwise(A, pw_0_attributes);
            A_after_pw_0->set_data_type(fe::DataType_t::BFLOAT16);

            C = graph.matmul(A_after_pw_0, B, matmul_attributes);
        } else if (use_bias) {
            // Create Bias vector
            auto Bias_attributes = fe::graph::Tensor_attributes()
                                       .set_name("Bias")
                                       .set_dim({matmul_shape.b, matmul_shape.m, 1})
                                       .set_stride({matmul_shape.m, 1, 1})
                                       .set_data_type(fe::DataType_t::BFLOAT16);
            Bias = graph.tensor(Bias_attributes);

            // Add ADD operation
            auto pw_0_attributes = fe::graph::Pointwise_attributes()
                                       .set_name("pw0_Add")
                                       .set_mode(fe::PointwiseMode_t::ADD)
                                       .set_compute_data_type(fe::DataType_t::FLOAT);

            auto A_after_pw_0 = graph.pointwise(A, Bias, pw_0_attributes);
            A_after_pw_0->set_data_type(fe::DataType_t::BFLOAT16);

            C = graph.matmul(A_after_pw_0, B, matmul_attributes);
        } else {
            C = graph.matmul(A, B, matmul_attributes);
        }
        C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

        std::cout << graph << std::endl;
        auto status = graph.validate();
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Dynamic shapes not supported pre 9.4");
        }

        status = graph.build_operation_graph(handle);
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Kernel cache not supported pre 9.4");
        }

        REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph.check_support(handle).is_good());

        REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good());

        return std::make_tuple(graph, A, B, C, Bias);
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    for (int idx_shape = 0; idx_shape < matmul_shapes_count; idx_shape++) {
        auto [graph, A, B, C, Bias] = build_new_graph(handle, idx_shape);
        // Initialize input tensors
        Surface<half> A_gpu(max_a_volume, false);
        Surface<half> B_gpu(max_b_volume, false);
        Surface<float> C_gpu(max_c_volume, false);
        Surface<half> Bias_gpu(max_bias_volume, false);
        Surface<int8_t> workspace(graph.get_workspace_size(), false);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
        if (use_bias) {
            variant_pack = {{A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}, {Bias, Bias_gpu.devPtr}};
        } else {
            variant_pack = {{A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};
        }
        REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    }
}

TEST_CASE("Matmul dynamic shape", "[matmul][graph][dynamic_shape]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    matmul_dynamic_shapes(false, false);  // Matmul dynamic shape, no abs or bias
}

TEST_CASE("Abs + Matmul dynamic shape", "[matmul][graph][dynamic_shape]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    matmul_dynamic_shapes(true, false);  // Matmul with abs
}

TEST_CASE("Bias + Matmul dynamic shape", "[matmul][graph][dynamic_shape]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    matmul_dynamic_shapes(false, true);  // Matmul with bias
}

TEST_CASE("Matmul", "[matmul][graph]") {
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
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

    auto matmul_attributes = fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT);
    auto C                 = graph.matmul(A, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    std::cout << graph << std::endl;
    REQUIRE(graph.validate().is_good());

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    graph.deselect_engines({"eng4_"});
    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good());

    // Run cudnn graph
    Surface<float> C_gpu(b * m * n, false);
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
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

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    // Run cudnn graph
    Surface<float> C_gpu(b * m * n, false);
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
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

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

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
}

TEST_CASE("Matmul SBR Graph", "[matmul][graph]") {
    namespace fe = cudnn_frontend;

    if (cudnnGetVersion() < 8600) {
        SKIP("Test requires cuDNN version 8.6.0 or above");
        return;
    }

    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

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

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    Surface<half> x_tensor(4 * 16 * 64, false);
    Surface<half> w_tensor(4 * 64 * 32, false);
    Surface<half> s_tensor(4 * 16 * 32, false);
    Surface<half> b_tensor(4 * 16 * 32, false);
    Surface<half> y_tensor(4 * 16 * 32, false);

    auto [graph, A, B, bias, scale, O] = lookup_cache_or_build_graph(
        handle, x_tensor.devPtr, w_tensor.devPtr, s_tensor.devPtr, b_tensor.devPtr, y_tensor.devPtr);

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{A, x_tensor.devPtr},
                                                                                             {B, w_tensor.devPtr},
                                                                                             {scale, s_tensor.devPtr},
                                                                                             {bias, b_tensor.devPtr},
                                                                                             {O, y_tensor.devPtr}};
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("Matmul with restricted shared memory", "[matmul][graph]") {
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }
    namespace fe = cudnn_frontend;

    // matmul problem size
    int64_t const b = 1;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 32;

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

    auto matmul_attributes = fe::graph::Matmul_attributes().set_compute_data_type(fe::DataType_t::FLOAT);
    auto C                 = graph.matmul(A, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    std::cout << graph << std::endl;
    REQUIRE(graph.validate().is_good());

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    graph.deselect_shared_mem_greater_than(256 * 1024);
    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    // Run cudnn graph
    Surface<float> C_gpu(b * m * n, false);
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {B, B_gpu.devPtr}, {C, C_gpu.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}