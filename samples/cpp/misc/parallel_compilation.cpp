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
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "../../utils/helpers.h"

#include <cudnn_frontend.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

TEST_CASE("Parallel build", "[matmul][graph][parallel]") {
    SKIP(
        "Very long test turned off by default. Run /bin/samples --benchmark-samples 1  \"Parallel build\" after "
        "uncommenting this line.");
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
    Surface<half> C_gpu(b * m * n, false);

    int64_t a_uid = 0, b_uid = 1, c_uid = 2;

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    auto create_graph = [&]() -> fe::graph::Graph {
        // Make cudnn graph
        fe::graph::Graph graph{};

        // Create the two non-virtual input tensors A and B.
        // There are read from global memory.
        auto A_attributes = fe::graph::Tensor_attributes()
                                .set_name("A")
                                .set_dim({b, m, k})
                                .set_stride({m * k, k, 1})
                                .set_uid(a_uid)
                                .set_data_type(fe::DataType_t::BFLOAT16);
        auto A            = graph.tensor(A_attributes);
        auto B_attributes = fe::graph::Tensor_attributes()
                                .set_name("B")
                                .set_dim({b, k, n})
                                .set_stride({k * n, n, 1})
                                .set_uid(b_uid)
                                .set_data_type(fe::DataType_t::BFLOAT16);
        auto B = graph.tensor(B_attributes);

        auto matmul_attributes =
            fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
        auto C = graph.matmul(A, B, matmul_attributes);
        C->set_output(true).set_uid(c_uid).set_data_type(fe::DataType_t::BFLOAT16);

        REQUIRE(graph.validate().is_good());

        REQUIRE(graph.build_operation_graph(handle).is_good());

        REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

        graph.select_behavior_notes({fe::BehaviorNote_t::RUNTIME_COMPILATION});

        REQUIRE(graph.check_support(handle).is_good());

        return graph;
    };

    auto build = [](fe::graph::Graph &graph, cudnnHandle_t handle, int index) {
        auto status = graph.build_plan_at_index(handle, index);
    };

    BENCHMARK("BuildPlanPolicy_t::HEURISTICS_CHOICE") {
        fe::graph::Graph graph = create_graph();
        return graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good();
    };

    BENCHMARK("BuildPlanPolicy_t::ALL") {
        fe::graph::Graph graph = create_graph();
        return graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good();
    };

    BENCHMARK("build_plan_at_index::ALL::serial") {
        fe::graph::Graph graph = create_graph();
        auto plan_count        = graph.get_execution_plan_count();
        for (auto i = 0; i < plan_count; i++) {
            build(graph, handle, i);
        }
    };

    BENCHMARK("build_plan_at_index::ALL::parallel") {
        fe::graph::Graph graph = create_graph();
        auto plan_count        = graph.get_execution_plan_count();
        std::vector<std::thread> builders;
        for (auto i = 0; i < plan_count; i++) {
            builders.emplace_back(std::thread{build, std::reference_wrapper<fe::graph::Graph>(graph), handle, i});
        }
        for (auto &builder : builders) {
            builder.join();
        }
    };

    {
        auto input = GENERATE(range(2, 11));

        BENCHMARK("build_plan_at_index::ALL::parallel_" + std::to_string(input)) {
            fe::graph::Graph graph = create_graph();
            auto plan_count = input < graph.get_execution_plan_count() ? input : graph.get_execution_plan_count();
            std::vector<std::thread> builders;
            for (auto i = 0; i < plan_count; i++) {
                builders.emplace_back(std::thread{build, std::reference_wrapper<fe::graph::Graph>(graph), handle, i});
            }
            for (auto &builder : builders) {
                builder.join();
            }
        };
    }

    checkCudnnErr(cudnnDestroy(handle));
}