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
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Matmul custom plan", "[matmul][graph][autotuning]") {
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }

    if (cudnnGetVersion() < 90300) {
        SKIP("Test requires cudnn 9.3.0 or above");
        return;
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

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

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

        return graph;
    };

    auto graph = create_graph();

    int64_t engine_count;
    REQUIRE(graph.get_engine_count(engine_count).is_good());

    for (int64_t id = 0; id < engine_count; id++) {
        std::vector<fe::Knob> knobs;

        // It might happen that an engine is not supported.
        auto status = graph.get_knobs_for_engine(id, knobs);
        if (status.get_code() != fe::error_code_t::OK) {
            continue;
        }

        std::unordered_map<fe::KnobType_t, int64_t> knob_map;
        for (auto &knob : knobs) {
            knob_map[knob.type] = knob.minValue + knob.stride;
        }

        // It might happen that the knobs are not supported.
        status = graph.create_execution_plan(id, knob_map);
        if (status.get_code() != fe::error_code_t::OK) {
            continue;
        }
    }

    REQUIRE(graph.check_support().is_good());

    auto plan_count = graph.get_execution_plan_count();
    std::cout << "Graph has " << plan_count << " plan candidates." << std::endl;

    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::ALL).is_good());

    std::unordered_map<int64_t, void *> variant_pack = {
        {a_uid, A_gpu.devPtr}, {b_uid, B_gpu.devPtr}, {c_uid, C_gpu.devPtr}};

    Surface<int8_t> workspace1(graph.get_workspace_size_plan_at_index(0), false);
    REQUIRE(graph.execute_plan_at_index(handle, variant_pack, workspace1.devPtr, 0).is_good());

    Surface<int8_t> workspace2(graph.get_workspace_size_plan_at_index(1), false);
    REQUIRE(graph.execute_plan_at_index(handle, variant_pack, workspace2.devPtr, 1).is_good());
}