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

struct InspectableGraph : fe::graph::Graph {
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
