/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cudnn_frontend.h>

#include <cstring>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

constexpr int64_t B = 2, H = 4, S = 128, D = 64, W = D + 8, START = 8;

std::vector<int64_t>
bshd_strides() {
    return {S * H * W, W, H * W, 1};
}

void *
device_halves(unsigned seed) {
    std::vector<__half> host(B * H * S * W);
    for (size_t i = 0; i < host.size(); i++) {
        seed    = seed * 1664525u + 1013904223u;
        host[i] = __float2half(static_cast<float>((seed >> 8) % 2001) / 1000.0f - 1.0f);
    }
    void *ptr = nullptr;
    REQUIRE(cudaMalloc(&ptr, host.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(ptr, host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice) == cudaSuccess);
    return ptr;
}

std::vector<__half>
to_host(void *ptr) {
    std::vector<__half> host(B * H * S * W);
    REQUIRE(cudaMemcpy(host.data(), ptr, host.size() * sizeof(__half), cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
}

using UidMap = std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *>;

void
skip_unless_legacy_slice() {
    if (fe::detail::get_backend_version() >= 92200 || fe::detail::get_compiled_version() >= 92200) {
        SKIP("Slice is lowered by pointer offset only before cuDNN 9.22");
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        SKIP("needs a CUDA GPU");
    }
}

// SDPA whose Q is a Slice of q_full (uid 1); K, V and O are uids 2, 3 and 4. Returns Q, or nullptr when no native
// plan serves the graph.
std::shared_ptr<fe::graph::Tensor_attributes>
build_slice_sdpa(fe::graph::Graph &graph, cudnnHandle_t handle) {
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto q_full = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("q_full")
                                   .set_uid(1)
                                   .set_dim({B, H, S, W})
                                   .set_stride(bshd_strides())
                                   .set_data_type(fe::DataType_t::HALF));
    std::vector<std::pair<int64_t, int64_t>> const slices{{0, B}, {0, H}, {0, S}, {START, START + D}};
    auto q = graph.slice(q_full, fe::graph::Slice_attributes().set_slices(slices));
    q->set_data_type(fe::DataType_t::HALF);
    auto k = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("k")
                              .set_uid(2)
                              .set_dim({B, H, S, D})
                              .set_stride(bshd_strides())
                              .set_data_type(fe::DataType_t::HALF));
    auto v = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("v")
                              .set_uid(3)
                              .set_dim({B, H, S, D})
                              .set_stride(bshd_strides())
                              .set_data_type(fe::DataType_t::HALF));
    auto [o, stats] =
        graph.sdpa(q, k, v, fe::graph::SDPA_attributes().set_generate_stats(false).set_attn_scale(0.125f));
    (void)stats;
    o->set_output(true).set_uid(4).set_dim({B, H, S, D}).set_stride(bshd_strides());

    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
    if (graph.check_support().is_bad() || graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_bad()) {
        return nullptr;
    }
    return q;
}

// Launches the CUDA graph and returns O. The output buffer is zeroed first so a stale result cannot pass.
std::vector<__half>
replay(cudaGraph_t cuda_graph, void *o_ptr) {
    REQUIRE(cudaMemset(o_ptr, 0, B * H * S * W * sizeof(__half)) == cudaSuccess);
    cudaGraphExec_t graph_exec;
    REQUIRE(cudaGraphInstantiate(&graph_exec, cuda_graph, 0) == cudaSuccess);
    REQUIRE(cudaGraphLaunch(graph_exec, 0) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cudaGraphExecDestroy(graph_exec);
    return to_host(o_ptr);
}

}  // namespace

// #1572: populate_cuda_graph() must not write the derived Slice destination into the caller's uid map, or a
// reused map's update_cuda_graph() keeps the old source's address.
TEST_CASE("CUDA graph update re-derives a legacy Slice binding from a reused uid map", "[cuda_graph][slice]") {
    skip_unless_legacy_slice();
    cudnnHandle_t handle;
    REQUIRE(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);
    fe::graph::Graph graph;
    auto q = build_slice_sdpa(graph, handle);
    if (q == nullptr) {
        cudnnDestroy(handle);
        SKIP("no native plan serves this graph");
    }

    void *first = device_halves(1), *second = device_halves(2), *k_ptr = device_halves(3), *v_ptr = device_halves(4);
    void *o_ptr            = device_halves(5);
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    void *workspace = nullptr;
    REQUIRE(cudaMalloc(&workspace, workspace_size > 0 ? workspace_size : 1) == cudaSuccess);

    // Reference: a graph populated from a fresh map bound to the second source.
    UidMap fresh{{1, second}, {2, k_ptr}, {3, v_ptr}, {4, o_ptr}};
    cudaGraph_t reference_graph, cuda_graph;
    REQUIRE(cudaGraphCreate(&reference_graph, 0) == cudaSuccess);
    REQUIRE(graph.populate_cuda_graph(handle, fresh, workspace, reference_graph).is_good());
    auto const reference         = replay(reference_graph, o_ptr);
    auto const from_first_source = [&] {
        UidMap once{{1, first}, {2, k_ptr}, {3, v_ptr}, {4, o_ptr}};
        cudaGraph_t first_graph;
        REQUIRE(cudaGraphCreate(&first_graph, 0) == cudaSuccess);
        REQUIRE(graph.populate_cuda_graph(handle, once, workspace, first_graph).is_good());
        auto out = replay(first_graph, o_ptr);
        cudaGraphDestroy(first_graph);
        return out;
    }();
    REQUIRE(std::memcmp(from_first_source.data(), reference.data(), reference.size() * sizeof(__half)) != 0);

    // Populate with the first source, then update with the same map object, changing only the source.
    UidMap bindings{{1, first}, {2, k_ptr}, {3, v_ptr}, {4, o_ptr}};
    REQUIRE(cudaGraphCreate(&cuda_graph, 0) == cudaSuccess);
    REQUIRE(graph.populate_cuda_graph(handle, bindings, workspace, cuda_graph).is_good());
    CHECK(bindings.count(q->get_uid()) == 0);
    bindings[1] = second;
    REQUIRE(graph.update_cuda_graph(handle, bindings, workspace, cuda_graph).is_good());
    auto const replayed = replay(cuda_graph, o_ptr);
    CHECK(std::memcmp(replayed.data(), reference.data(), reference.size() * sizeof(__half)) == 0);

    cudaGraphDestroy(reference_graph);
    cudaGraphDestroy(cuda_graph);
    for (void *ptr : {first, second, k_ptr, v_ptr, o_ptr, workspace}) {
        cudaFree(ptr);
    }
    cudnnDestroy(handle);
}

// The backend graph does not consume a legacy Slice source; execute() must not hand its uid to the backend, which
// rejects an unknown uid (varPackUidCount > expected_uid_count).
TEST_CASE("execute() runs a legacy Slice graph whose source the backend does not consume", "[slice][execute]") {
    skip_unless_legacy_slice();
    cudnnHandle_t handle;
    REQUIRE(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);
    fe::graph::Graph graph;
    if (build_slice_sdpa(graph, handle) == nullptr) {
        cudnnDestroy(handle);
        SKIP("no native plan serves this graph");
    }

    void *q_full = device_halves(1), *k_ptr = device_halves(3), *v_ptr = device_halves(4), *o_ptr = device_halves(5);
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    void *workspace = nullptr;
    REQUIRE(cudaMalloc(&workspace, workspace_size > 0 ? workspace_size : 1) == cudaSuccess);
    UidMap bindings{{1, q_full}, {2, k_ptr}, {3, v_ptr}, {4, o_ptr}};

    cudaGraph_t reference_graph;
    REQUIRE(cudaGraphCreate(&reference_graph, 0) == cudaSuccess);
    REQUIRE(graph.populate_cuda_graph(handle, bindings, workspace, reference_graph).is_good());
    auto const reference = replay(reference_graph, o_ptr);

    REQUIRE(cudaMemset(o_ptr, 0, B * H * S * W * sizeof(__half)) == cudaSuccess);
    auto status = graph.execute(handle, bindings, workspace);
    INFO(status.get_message());
    REQUIRE(status.is_good());
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(std::memcmp(to_host(o_ptr).data(), reference.data(), reference.size() * sizeof(__half)) == 0);

    // The sorted-pointer overload still takes the source as a user slot.
    std::vector<void *> sorted_ptrs;
    for (auto uid : graph.get_variant_pack_uids_sorted()) {
        REQUIRE(bindings.count(uid) == 1);
        sorted_ptrs.push_back(bindings[uid]);
    }
    REQUIRE(cudaMemset(o_ptr, 0, B * H * S * W * sizeof(__half)) == cudaSuccess);
    REQUIRE(graph.execute(handle, sorted_ptrs.data(), (int)sorted_ptrs.size(), workspace).is_good());
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(std::memcmp(to_host(o_ptr).data(), reference.data(), reference.size() * sizeof(__half)) == 0);

    cudaGraphDestroy(reference_graph);
    for (void *ptr : {q_full, k_ptr, v_ptr, o_ptr, workspace}) {
        cudaFree(ptr);
    }
    cudnnDestroy(handle);
}
