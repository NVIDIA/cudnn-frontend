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
#include "../../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Matmul autotuning", "[matmul][graph][autotuning]") {
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

        graph.deselect_workspace_greater_than(0);

        REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

        graph.deselect_workspace_greater_than(1024 * 1024);

        REQUIRE(graph.check_support(handle).is_good());

        return graph;
    };

    auto graph = create_graph();

    auto plan_count = graph.get_execution_plan_count();
    std::cout << "Graph has " << plan_count << " plan candidates." << std::endl;

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good());

    std::unordered_map<int64_t, void *> variant_pack = {
        {a_uid, A_gpu.devPtr}, {b_uid, B_gpu.devPtr}, {c_uid, C_gpu.devPtr}};

    auto autotune = [&]() -> int64_t {
        const int iter_count = 10;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();

        cudaStream_t stream = nullptr;
        cudnnGetStream(handle, &stream);

        std::vector<float> execution_times;
        execution_times.resize(plan_count, 10.0f);  // Some arbitrary high time

        int64_t workspace_size = 0;
        for (auto i = 0; i < plan_count; i++) {
            workspace_size = std::max(workspace_size, graph.get_workspace_size_plan_at_index(i));
        }

        Surface<int8_t> workspace(workspace_size, false);

        for (auto i = 0; i < plan_count; i++) {
            float time_ms = 0.0f;

            auto warmup_status = graph.execute_plan_at_index(handle, variant_pack, workspace.devPtr, i);

            if (warmup_status.is_bad()) {
                std::cout << "Plan at index " << i << " failed execution " << warmup_status.get_message() << std::endl;
                continue;
            }
            cudaDeviceSynchronize();

            cudaEventRecord(start, stream);
            for (int iter = 0; iter < iter_count; iter++) {
                auto status = graph.execute_plan_at_index(handle, variant_pack, workspace.devPtr, i);
                (void)status;
            }
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_ms, start, stop);

            std::cout << "Plan at index " << i << " took " << time_ms / iter_count << " ms." << std::endl;
            execution_times[i] = time_ms / iter_count;
        }

        return std::distance(std::begin(execution_times),
                             std::min_element(std::begin(execution_times), std::end(execution_times)));
    };
    // Run cudnn graph

    auto candidate_index = autotune();

    std::cout << "Successful candidate is at index " << candidate_index << std::endl;

    REQUIRE(graph.build_plan_at_index(handle, candidate_index).is_good());

    Surface<int8_t> workspace(graph.get_workspace_size_plan_at_index(candidate_index), false);

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    checkCudnnErr(cudnnDestroy(handle));
}