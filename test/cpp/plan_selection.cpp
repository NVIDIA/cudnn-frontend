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

struct StubOssNormEngine : fe::experimental::IOssNormEngine {
    bool build_succeeds;
    explicit StubOssNormEngine(bool succeeds) : build_succeeds(succeeds) {}
    fe::error_t
    check_support(fe::experimental::NormSiluShape_t, int) override {
        return {fe::error_code_t::OK, ""};
    }
    fe::error_t
    build() override {
        return build_succeeds
                   ? fe::error_t{fe::error_code_t::OK, ""}
                   : fe::error_t{fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "test build failure"};
    }
    fe::error_t
    execute(void*,
            void*,
            void*,
            void*,
            int,
            int,
            float,
            void*,
            int,
            cudaStream_t,
            fe::experimental::RmsNormSiluExtraParams const&) override {
        FAIL("The selection test must not execute the stub engine");
        return {fe::error_code_t::GRAPH_EXECUTION_FAILED, "test engine has no kernel"};
    }
    int64_t
    get_workspace_size() const override {
        return 0;
    }
};

void
register_supported_oss_engine(fe::graph::Execution_plan_list& plans, bool build_succeeds) {
    plans.set_oss_rms_norm_silu_engine(std::make_shared<StubOssNormEngine>(build_succeeds));
    REQUIRE(plans.check_oss_rms_norm_silu_support(100).is_good());
}

struct InspectableGraph : fe::graph::Graph {
    void
    register_oss_engine(bool build_succeeds) {
        register_supported_oss_engine(plans, build_succeeds);
    }

    int64_t
    selected_index() const {
        return plans.candidate;
    }
    bool
    is_built(size_t i) const {
        return plans.execution_plans.at(i) != nullptr;
    }
};

std::shared_ptr<InspectableGraph>
make_matmul_graph(cudnnHandle_t handle) {
    auto graph = std::make_shared<InspectableGraph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto a = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 16, 64}).set_stride({1024, 64, 1}));
    auto b = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 64, 32}).set_stride({2048, 32, 1}));
    graph->matmul(a, b, fe::graph::Matmul_attributes())->set_output(true);
    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    return graph;
}
}  // namespace

TEST_CASE("Building all plans preserves the first selected candidate", "[graph][plan_selection]") {
    Handle handle;
    auto source = make_matmul_graph(handle.value);
    REQUIRE(source->create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(source->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
    int64_t engine = -1;
    std::unordered_map<fe::KnobType_t, int64_t> knobs;
    REQUIRE(source->get_engine_and_knobs_at_index(source->selected_index(), engine, knobs).is_good());

    // Two known-buildable configs guarantee multiple successes without depending
    // on a particular GPU's heuristic count, engine IDs, or knob choices.
    auto graph = make_matmul_graph(handle.value);
    REQUIRE(graph->create_execution_plan(engine, knobs).is_good());
    REQUIRE(graph->create_execution_plan(engine, knobs).is_good());
    REQUIRE(graph->get_execution_plan_count() == 2);

    bool expect_rejection = false;
    int64_t expected      = 0;
    SECTION("a previous selection must still pass filters") {
        REQUIRE(graph->build_plan_at_index(0).is_good());
        std::string name;
        REQUIRE(graph->get_plan_name_at_index(0, name).is_good());
        graph->deselect_engines({name});
        expect_rejection = true;
    }

    SECTION("an OSS build failure falls back to a built native plan") { graph->register_oss_engine(false); }
    SECTION("a successfully built OSS candidate stays selected") {
        graph->register_oss_engine(true);
        expected = fe::graph::Execution_plan_list::OSS_RMS_NORM_SILU_ENGINE_CANDIDATE;
    }

    SECTION("no previous selection") { REQUIRE(graph->selected_index() == -1); }
    SECTION("explicit first plan") { REQUIRE(graph->build_plan_at_index(0).is_good()); }
    SECTION("explicit second plan") {
        REQUIRE(graph->build_plan_at_index(1).is_good());
        expected = 1;
    }

    auto status = graph->build_plans(fe::BuildPlanPolicy_t::ALL);
    if (expect_rejection) {
        REQUIRE(status.is_bad());
        REQUIRE(graph->selected_index() == -1);
    } else {
        REQUIRE(status.is_good());
        REQUIRE(graph->is_built(0));
        REQUIRE(graph->is_built(1));
        REQUIRE(graph->selected_index() == expected);
        REQUIRE(graph->build_plans(fe::BuildPlanPolicy_t::ALL).is_good());
        REQUIRE(graph->selected_index() == expected);
    }
}

TEST_CASE("An unbuilt OSS candidate cannot make build-all succeed", "[graph][plan_selection]") {
    fe::graph::Execution_plan_list plans;
    register_supported_oss_engine(plans, false);
    REQUIRE(plans.build_oss_rms_norm_silu_engine().is_bad());
    REQUIRE(plans.build_plans(fe::BuildPlanPolicy_t::ALL, false).is_bad());
    REQUIRE(plans.candidate == -1);
}
