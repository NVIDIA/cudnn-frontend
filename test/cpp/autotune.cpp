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

struct DeviceBuffer {
    void* value = nullptr;
    explicit DeviceBuffer(size_t bytes) { REQUIRE(cudaMalloc(&value, std::max(bytes, size_t{1})) == cudaSuccess); }
    ~DeviceBuffer() { cudaFree(value); }
};

struct PlanMetadata {
    int64_t engine = -1;
    std::unordered_map<fe::KnobType_t, int64_t> knobs;
    std::string name;
    std::vector<fe::NumericalNote_t> numeric;
    std::vector<fe::BehaviorNote_t> behavior;
};

struct InspectableGraph : fe::graph::Graph {
    PlanMetadata
    metadata(int64_t index) const {
        PlanMetadata result;
        REQUIRE(get_engine_and_knobs_at_index(index, result.engine, result.knobs).is_good());
        REQUIRE(get_plan_name_at_index(index, result.name).is_good());
        result.numeric = plans.numeric_notes.at(index);
        REQUIRE(get_behavior_notes_for_plan_at_index(index, result.behavior).is_good());
        return result;
    }

    auto const&
    built_plans() const {
        return plans.execution_plans;
    }
    int64_t
    selected_index() const {
        return plans.candidate;
    }
};

std::shared_ptr<InspectableGraph>
make_matmul(cudnnHandle_t handle) {
    auto graph = std::make_shared<InspectableGraph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto a = graph->tensor(fe::graph::Tensor_attributes().set_uid(1).set_dim({1, 16, 64}).set_stride({1024, 64, 1}));
    auto b = graph->tensor(fe::graph::Tensor_attributes().set_uid(2).set_dim({1, 64, 32}).set_stride({2048, 32, 1}));
    graph->matmul(a, b, fe::graph::Matmul_attributes())->set_uid(3).set_output(true);
    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    return graph;
}

std::shared_ptr<InspectableGraph>
make_three_candidates(cudnnHandle_t handle) {
    auto source = make_matmul(handle);
    REQUIRE(source->create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(source->build_plans().is_good());
    int64_t engine = -1;
    std::unordered_map<fe::KnobType_t, int64_t> knobs;
    REQUIRE(source->get_engine_and_knobs_at_index(source->selected_index(), engine, knobs).is_good());
    auto graph = make_matmul(handle);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(graph->create_execution_plan(engine, knobs).is_good());
    }
    return graph;
}

struct Bindings {
    DeviceBuffer a{2048}, b{4096}, c{1024};
    std::unordered_map<int64_t, void*> pointers{{1, a.value}, {2, b.value}, {3, c.value}};
    Bindings() {
        // Exact FP16 ones: every output of the 16x64 @ 64x32 matmul must be 64.
        std::vector<uint16_t> ones(2048, 0x3c00);
        REQUIRE(cudaMemcpy(a.value, ones.data(), 2048, cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(b.value, ones.data(), 4096, cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemset(c.value, 0x7f, 1024) == cudaSuccess);
    }
    void
    check_output() const {
        std::vector<uint16_t> output(512);
        REQUIRE(cudaMemcpy(output.data(), c.value, 1024, cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(std::all_of(output.begin(), output.end(), [](uint16_t value) { return value == 0x5400; }));
    }
};
}  // namespace

TEST_CASE("Failed graph autotune preserves built plans and selection", "[graph][autotune]") {
    Handle handle;
    auto graph = make_three_candidates(handle.value);
    REQUIRE(graph->build_plan_at_index(0).is_good());
    REQUIRE(graph->build_plan_at_index(2).is_good());
    auto const original_plans = graph->built_plans();
    REQUIRE(graph->selected_index() == 2);
    DeviceBuffer workspace(static_cast<size_t>(graph->get_autotune_workspace_size()));
    std::unordered_map<int64_t, void*> missing_bindings;

    auto status = graph->autotune(handle.value, missing_bindings, workspace.value);
    CHECK(status.get_code() == fe::error_code_t::GRAPH_EXECUTION_FAILED);
    CHECK(status.get_message().find("No execution plans were successfully timed") != std::string::npos);
    CHECK(status.get_message().find("Plan at index 0: Uid 1 not found in variant pack.") != std::string::npos);
    CHECK(status.get_message().find("Plan at index 2: Uid 1 not found in variant pack.") != std::string::npos);
    REQUIRE(graph->built_plans() == original_plans);
    REQUIRE(graph->selected_index() == 2);

    // Correct bindings can execute the previously selected plan after a failed tune.
    Bindings bindings;
    REQUIRE(graph->execute(handle.value, bindings.pointers, workspace.value).is_good());
    bindings.check_output();

    // Retrying with valid bindings still tunes the two built plans, skipping the null slot.
    REQUIRE(graph->autotune(handle.value, bindings.pointers, workspace.value).is_good());
    REQUIRE(graph->get_execution_plan_count() == 2);
    REQUIRE(graph->selected_index() == 0);
    REQUIRE(graph->execute(handle.value, bindings.pointers, workspace.value).is_good());
    bindings.check_output();
}

TEST_CASE("Graph autotune requires a built candidate", "[graph][autotune]") {
    Handle handle;
    auto graph = make_matmul(handle.value);
    SECTION("no plans") {}
    SECTION("all plans unbuilt") { REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good()); }
    auto const original_plans     = graph->built_plans();
    auto const original_candidate = graph->selected_index();
    std::unordered_map<int64_t, void*> bindings;
    auto status = graph->autotune(handle.value, bindings, nullptr);
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_EXECUTION_FAILED);
    REQUIRE(graph->built_plans() == original_plans);
    REQUIRE(graph->selected_index() == original_candidate);
}

TEST_CASE("Graph autotune keeps plan identity through sorting and pruning", "[graph][autotune]") {
    Handle handle;
    auto graph = make_matmul(handle.value);
    REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph->build_plans(fe::BuildPlanPolicy_t::ALL).is_good());

    SECTION("all built heuristic plans") {}
    SECTION("a filtered prefix and unbuilt suffix are removed") {
        // Pick two actually buildable configs with distinct names, without pinning
        // engine IDs, knob choices, or which plan should win a timing comparison.
        std::vector<PlanMetadata> configs;
        for (int64_t i = 0; i < graph->get_execution_plan_count(); ++i) {
            if (graph->built_plans()[i] == nullptr) continue;
            auto metadata = graph->metadata(i);
            if (configs.empty() || metadata.name.find(configs.front().name) == std::string::npos) {
                configs.push_back(std::move(metadata));
            }
            if (configs.size() == 2) break;
        }
        if (configs.size() != 2) SKIP("Requires two buildable configs with distinct engine names");
        graph = make_matmul(handle.value);
        for (auto const& config : {configs[0], configs[1], configs[0]}) {
            REQUIRE(graph->create_execution_plan(config.engine, config.knobs).is_good());
        }
        graph->deselect_engines({configs[0].name});
        REQUIRE(graph->build_plan_at_index(0).is_bad());
        REQUIRE(graph->build_plan_at_index(1).is_good());
        REQUIRE(graph->built_plans()[0] == nullptr);
        REQUIRE(graph->built_plans()[2] == nullptr);
    }

    std::unordered_map<fe::ExecutionPlan const*, PlanMetadata> original;
    for (int64_t i = 0; i < graph->get_execution_plan_count(); ++i) {
        if (graph->built_plans()[i]) original.emplace(graph->built_plans()[i].get(), graph->metadata(i));
    }
    REQUIRE_FALSE(original.empty());
    Bindings bindings;
    DeviceBuffer workspace(static_cast<size_t>(graph->get_autotune_workspace_size()));
    REQUIRE(graph->autotune(handle.value, bindings.pointers, workspace.value).is_good());
    REQUIRE(graph->selected_index() == 0);
    REQUIRE(graph->get_execution_plan_count() == static_cast<int64_t>(original.size()));
    for (int64_t i = 0; i < graph->get_execution_plan_count(); ++i) {
        auto const& expected = original.at(graph->built_plans()[i].get());
        auto const actual    = graph->metadata(i);
        CHECK(actual.engine == expected.engine);
        CHECK(actual.knobs == expected.knobs);
        CHECK(actual.name == expected.name);
        CHECK(actual.numeric == expected.numeric);
        CHECK(actual.behavior == expected.behavior);
        // A pruned rejected prefix must not leave a stale barred flag on the winner.
        CHECK(graph->build_plan_at_index(i).is_good());
    }
    REQUIRE(graph->build_plan_at_index(0).is_good());
    REQUIRE(graph->execute(handle.value, bindings.pointers, workspace.value).is_good());
    bindings.check_output();

    // The saved winner must reconstruct that same config on a fresh graph.
    auto const expected_winner = original.at(graph->built_plans()[0].get());
    auto const winner          = graph->metadata(0);
    auto replay                = make_matmul(handle.value);
    REQUIRE(replay->create_execution_plan(winner.engine, winner.knobs).is_good());
    REQUIRE(replay->build_plan_at_index(0).is_good());
    CHECK(replay->metadata(0).engine == expected_winner.engine);
    CHECK(replay->metadata(0).knobs == expected_winner.knobs);
    DeviceBuffer replay_workspace(static_cast<size_t>(replay->get_workspace_size()));
    REQUIRE(cudaMemset(bindings.c.value, 0x7f, 1024) == cudaSuccess);
    REQUIRE(replay->execute(handle.value, bindings.pointers, replay_workspace.value).is_good());
    bindings.check_output();

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    // Serialized graphs have behavior notes but no engine-config/numeric-note list.
    std::vector<uint8_t> blob;
    REQUIRE(graph->serialize(blob).is_good());
    InspectableGraph reloaded;
    REQUIRE(reloaded.deserialize(handle.value, blob, false, false).is_good());
    REQUIRE(reloaded.autotune(handle.value, bindings.pointers, workspace.value).is_good());
    std::vector<fe::BehaviorNote_t> reloaded_notes;
    REQUIRE(reloaded.get_behavior_notes(reloaded_notes).is_good());
    CHECK(reloaded_notes == expected_winner.behavior);
    REQUIRE(cudaMemset(bindings.c.value, 0x7f, 1024) == cudaSuccess);
    REQUIRE(reloaded.execute(handle.value, bindings.pointers, workspace.value).is_good());
    bindings.check_output();
#endif

    // Appending after compaction must add exactly one config, with its own notes.
    auto const count = graph->get_execution_plan_count();
    REQUIRE(graph->create_execution_plan(expected_winner.engine, expected_winner.knobs).is_good());
    CHECK(graph->get_execution_plan_count() == count + 1);
    auto const appended = graph->metadata(count);
    CHECK(appended.engine == expected_winner.engine);
    CHECK(appended.knobs == expected_winner.knobs);
    CHECK(appended.numeric == expected_winner.numeric);
    CHECK(appended.behavior == expected_winner.behavior);
    REQUIRE(graph->build_plan_at_index(count).is_good());
    REQUIRE(graph->execute(handle.value, bindings.pointers, workspace.value).is_good());
    bindings.check_output();
}
