/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cudnn_frontend.h>

namespace {
namespace fe = cudnn_frontend;

// Build a simple b x m x k @ b x k x n matmul graph (shared handle).
std::shared_ptr<fe::graph::Graph>
make_matmul_graph(cudnnHandle_t handle) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A =
        graph->tensor(fe::graph::Tensor_attributes().set_name("A").set_dim({4, 16, 64}).set_stride({16 * 64, 64, 1}));
    auto B =
        graph->tensor(fe::graph::Tensor_attributes().set_name("B").set_dim({4, 64, 32}).set_stride({64 * 32, 32, 1}));
    auto C = graph->matmul(A, B, fe::graph::Matmul_attributes().set_name("matmul"));
    C->set_output(true);

    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    return graph;
}
}  // namespace

TEST_CASE("get_engine_and_knobs_at_index round-trips via create_execution_plan", "[graph][knobs]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    auto graph = make_matmul_graph(handle);
    REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph->check_support().is_good());
    REQUIRE(graph->build_plans(fe::BuildPlanPolicy_t::ALL).is_good());

    auto const count = graph->get_execution_plan_count();
    REQUIRE(count > 0);

    // Out-of-range indices must error rather than read OOB.
    int64_t engine_id = -1;
    std::unordered_map<fe::KnobType_t, int64_t> knobs;
    REQUIRE(graph->get_engine_and_knobs_at_index(-1, engine_id, knobs).is_bad());
    REQUIRE(graph->get_engine_and_knobs_at_index(count, engine_id, knobs).is_bad());

    // For every plan: the reported (engine, knobs) must feed straight back into
    // create_execution_plan -- i.e. every backend knob type the engine uses is
    // representable as a KnobType_t. Building the pinned plan is best-effort
    // (it can fail for environment reasons, e.g. a ptxas older than the
    // engine's target); when it succeeds it must reproduce the same plan.
    int64_t round_tripped = 0;
    for (int64_t i = 0; i < count; i++) {
        std::string name;
        REQUIRE(graph->get_plan_name_at_index(i, name).is_good());

        engine_id = -1;
        knobs.clear();
        REQUIRE(graph->get_engine_and_knobs_at_index(i, engine_id, knobs).is_good());
        REQUIRE(engine_id >= 0);

        auto pinned = make_matmul_graph(handle);
        REQUIRE(pinned->create_execution_plan(engine_id, knobs).is_good());

        if (pinned->build_plans().is_good()) {
            REQUIRE(pinned->get_execution_plan_count() == 1);

            // Structured identity (engine + knob map) must match.
            int64_t pinned_engine = -1;
            std::unordered_map<fe::KnobType_t, int64_t> pinned_knobs;
            REQUIRE(pinned->get_engine_and_knobs_at_index(0, pinned_engine, pinned_knobs).is_good());
            REQUIRE(pinned_engine == engine_id);
            REQUIRE(pinned_knobs == knobs);

            // The plan-name tag is canonical (get_engine_tag sorts knobs by
            // type), so it must match too despite the two configs storing
            // knob choices in different orders.
            std::string pinned_name;
            REQUIRE(pinned->get_plan_name_at_index(0, pinned_name).is_good());
            REQUIRE(pinned_name == name);

            round_tripped++;
        }
    }
    REQUIRE(round_tripped > 0);

    cudnnDestroy(handle);
}
