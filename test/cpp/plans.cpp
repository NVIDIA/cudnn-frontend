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
}  // namespace

TEST_CASE("Plan failures retain individual rejection reasons", "[graph][plans]") {
    Handle handle;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto a = graph.tensor(fe::graph::Tensor_attributes().set_dim({1, 16, 64}).set_stride({1024, 64, 1}));
    auto b = graph.tensor(fe::graph::Tensor_attributes().set_dim({1, 64, 32}).set_stride({2048, 32, 1}));
    graph.matmul(a, b, fe::graph::Matmul_attributes())->set_output(true);
    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.build_operation_graph(handle.value).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.get_execution_plan_count() > 0);

    std::vector<std::string> names;
    for (int64_t i = 0; i < graph.get_execution_plan_count(); ++i) {
        std::string name;
        REQUIRE(graph.get_plan_name_at_index(i, name).is_good());
        names.push_back(name);
    }
    graph.deselect_engines(names);

    fe::error_t status;
    SECTION("support check") { status = graph.check_support(); }
    SECTION("heuristics choice build") { status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE); }
    SECTION("build all") { status = graph.build_plans(fe::BuildPlanPolicy_t::ALL); }

    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
    auto const& message = status.get_message();
    INFO(message);
    for (size_t i = 0; i < names.size(); ++i) {
        REQUIRE(message.find("Deselecting execution plan with name " + names[i] + " at position " +
                             std::to_string(i)) != std::string::npos);
    }
}

TEST_CASE("Empty plan lists keep their summary errors", "[graph][plans]") {
    fe::graph::Execution_plan_list plans;
    auto status = plans.check_support();
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
    REQUIRE(status.get_message().find("No execution plans support the graph.") != std::string::npos);
    status = plans.build_plans(fe::BuildPlanPolicy_t::ALL, false);
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
    REQUIRE(status.get_message() == "[cudnn_frontend] Error: No valid execution plans built.");
}
