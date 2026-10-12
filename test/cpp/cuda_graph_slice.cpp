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

}  // namespace

// #1572: populate_cuda_graph() must not write the derived Slice destination into the caller's uid map, or a
// reused map's update_cuda_graph() keeps the old source's address.
TEST_CASE("CUDA graph update re-derives a legacy Slice binding from a reused uid map", "[cuda_graph][slice]") {
    if (fe::detail::get_backend_version() >= 92200 || fe::detail::get_compiled_version() >= 92200) {
        SKIP("Slice is lowered by pointer offset only before cuDNN 9.22");
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        SKIP("needs a CUDA GPU");
    }

    cudnnHandle_t handle;
    REQUIRE(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);

    fe::graph::Graph graph;
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
        cudnnDestroy(handle);
        SKIP("no native plan serves this graph");
    }

    void *first = device_halves(1), *second = device_halves(2), *k_ptr = device_halves(3), *v_ptr = device_halves(4);
    void *o_ptr            = device_halves(5);
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    void *workspace = nullptr;
    REQUIRE(cudaMalloc(&workspace, workspace_size > 0 ? workspace_size : 1) == cudaSuccess);

    // Replays the graph and returns O. The output buffer is zeroed first so a stale result cannot pass.
    auto replay = [&](cudaGraph_t cuda_graph) {
        REQUIRE(cudaMemset(o_ptr, 0, B * H * S * W * sizeof(__half)) == cudaSuccess);
        cudaGraphExec_t graph_exec;
        REQUIRE(cudaGraphInstantiate(&graph_exec, cuda_graph, 0) == cudaSuccess);
        REQUIRE(cudaGraphLaunch(graph_exec, 0) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        cudaGraphExecDestroy(graph_exec);
        return to_host(o_ptr);
    };

    // Reference: a graph populated from a fresh map bound to the second source.
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> fresh{
        {1, second}, {2, k_ptr}, {3, v_ptr}, {4, o_ptr}};
    cudaGraph_t reference_graph, cuda_graph;
    REQUIRE(cudaGraphCreate(&reference_graph, 0) == cudaSuccess);
    REQUIRE(graph.populate_cuda_graph(handle, fresh, workspace, reference_graph).is_good());
    auto const reference         = replay(reference_graph);
    auto const from_first_source = [&] {
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> once{
            {1, first}, {2, k_ptr}, {3, v_ptr}, {4, o_ptr}};
        cudaGraph_t first_graph;
        REQUIRE(cudaGraphCreate(&first_graph, 0) == cudaSuccess);
        REQUIRE(graph.populate_cuda_graph(handle, once, workspace, first_graph).is_good());
        auto out = replay(first_graph);
        cudaGraphDestroy(first_graph);
        return out;
    }();
    REQUIRE(std::memcmp(from_first_source.data(), reference.data(), reference.size() * sizeof(__half)) != 0);

    // Populate with the first source, then update with the same map object, changing only the source.
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> bindings{
        {1, first}, {2, k_ptr}, {3, v_ptr}, {4, o_ptr}};
    REQUIRE(cudaGraphCreate(&cuda_graph, 0) == cudaSuccess);
    REQUIRE(graph.populate_cuda_graph(handle, bindings, workspace, cuda_graph).is_good());
    CHECK(bindings.count(q->get_uid()) == 0);
    bindings[1] = second;
    REQUIRE(graph.update_cuda_graph(handle, bindings, workspace, cuda_graph).is_good());
    auto const replayed = replay(cuda_graph);
    CHECK(std::memcmp(replayed.data(), reference.data(), reference.size() * sizeof(__half)) == 0);

    cudaGraphDestroy(reference_graph);
    cudaGraphDestroy(cuda_graph);
    for (void *ptr : {first, second, k_ptr, v_ptr, o_ptr, workspace}) {
        cudaFree(ptr);
    }
    cudnnDestroy(handle);
}
