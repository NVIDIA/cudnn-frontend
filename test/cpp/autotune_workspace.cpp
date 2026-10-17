/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

namespace {
namespace fe = cudnn_frontend;

struct Handle {
    cudnnHandle_t value = nullptr;
    Handle() { REQUIRE(cudnnCreate(&value) == CUDNN_STATUS_SUCCESS); }
    ~Handle() { cudnnDestroy(value); }
};

std::shared_ptr<fe::graph::Graph>
make_matmul_graph(cudnnHandle_t handle) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto a = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 256, 512}).set_stride({131072, 512, 1}));
    auto b = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 512, 768}).set_stride({393216, 768, 1}));
    graph->matmul(a, b, fe::graph::Matmul_attributes())->set_output(true);
    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    return graph;
}
}  // namespace

TEST_CASE("Autotune workspace ignores unbuilt plans", "[graph][autotune_workspace]") {
    Handle handle;
    auto source = make_matmul_graph(handle.value);
    REQUIRE(source->create_execution_plans({fe::HeurMode_t::A}).is_good());

    // Discover one buildable configuration without depending on an engine ID.
    int64_t engine = -1;
    std::unordered_map<fe::KnobType_t, int64_t> knobs;
    for (int64_t i = 0; i < source->get_execution_plan_count(); ++i) {
        if (source->build_plan_at_index(i).is_good()) {
            REQUIRE(source->get_engine_and_knobs_at_index(i, engine, knobs).is_good());
            break;
        }
    }
    REQUIRE(engine >= 0);
    auto graph = make_matmul_graph(handle.value);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(graph->create_execution_plan(engine, knobs).is_good());
    }
    REQUIRE(graph->build_plan_at_index(1).is_good());
    auto const expected = graph->get_workspace_size_plan_at_index(1);
    REQUIRE(expected >= 0);

    // Unbuilt entries before and after a real plan must not be dereferenced.
    REQUIRE(graph->get_autotune_workspace_size() == expected);
    REQUIRE(graph->get_workspace_size() == expected);
    REQUIRE(graph->build_plans(fe::BuildPlanPolicy_t::ALL).is_good());
    REQUIRE(graph->get_autotune_workspace_size() == expected);
}

TEST_CASE("Unbuilt plan lists need no autotune workspace", "[graph][autotune_workspace]") {
    fe::graph::Execution_plan_list plans;
    REQUIRE(plans.get_autotune_workspace() == 0);
    plans.execution_plans.resize(3);
    REQUIRE(plans.get_autotune_workspace() == 0);
}
