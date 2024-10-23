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

#define A_UID 0
#define B_UID 1
#define C_UID 2
#define D_UID 3

std::shared_ptr<cudnn_frontend::graph::Graph>
create_graph(int64_t b, int64_t m, int64_t n, int64_t k, float scale_value) {
    //// Create the cudnn graph
    auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
    graph->set_io_data_type(cudnn_frontend::DataType_t::HALF)
        .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto A = graph->tensor(
        cudnn_frontend::graph::Tensor_attributes().set_dim({b, m, k}).set_stride({m * k, k, 1}).set_uid(A_UID));

    auto scale_options = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    auto S             = graph->pointwise(A, graph->tensor(scale_value), scale_options);
    S->set_data_type(cudnn_frontend::DataType_t::HALF);

    auto B = graph->tensor(
        cudnn_frontend::graph::Tensor_attributes().set_dim({b, k, n}).set_stride({n * k, n, 1}).set_uid(B_UID));
    auto T = graph->matmul(S, B, cudnn_frontend::graph::Matmul_attributes());

    auto C           = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                               .set_dim({1, 1, 1})
                               .set_stride({1, 1, 1})
                               .set_is_pass_by_value(true)
                               .set_uid(C_UID));
    auto add_options = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::ADD);
    auto D           = graph->pointwise(T, C, add_options);
    D->set_output(true).set_uid(D_UID);
    return graph;
}

TEST_CASE("Cuda graphs with matmul add", "[cudagraph][graph]") {
    //// Main graph
    // This example shows how to add a cudnn cuda graph to an already existing cuda graph.
    cudaGraph_t main_cuda_graph;
    cudaGraphCreate(&main_cuda_graph, 0);

    // Create any FE graph that you want to create a cuda graph for
    int64_t b = 8, m = 32, n = 16, k = 8;
    float scale_value = .5f;
    auto graph        = create_graph(b, m, n, k, scale_value);

    // Create the execution plan, as that is needed to populate cuda graph with cudnn kernels
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // Validare the graph and lower the FE graph to BE graph
    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    REQUIRE(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());

    // Make sure the selected executino plan supports cuda graph
    graph->select_behavior_notes({cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
    auto status = graph->check_support(handle);
    if (cudnn_frontend::detail::get_backend_version() >= 90500) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.is_bad());
        SKIP("cudnn versions 9.5 and earlier don't support behavior note of SUPPORTS_CUDA_GRAPH_NATIVE_API.");
    }
    REQUIRE(graph->build_plans(handle).is_good());

    //// Populate an exisiting cuda graph with cudnn's cuda graph
    cudaGraph_t cudnn_cuda_graph;

    // Initialize the cudnn cuda graph.
    // The responsibility to destroy is on the user.
    cudaGraphCreate(&cudnn_cuda_graph, 0);  // 0 is just what the API says to pass

    Surface<int8_t> workspace(graph->get_workspace_size(), false);

    half starter_value = __float2half(1.f);
    half bias_value    = __float2half(2.f);
    Surface<half> a_gpu(b * m * k, false, starter_value);
    Surface<half> b_gpu(b * k * n, false, starter_value);
    Surface<half> d_gpu(b * m * n, false);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {A_UID, a_gpu.devPtr}, {B_UID, b_gpu.devPtr}, {C_UID, &bias_value}, {D_UID, d_gpu.devPtr}};

    REQUIRE(graph->populate_cuda_graph(handle, variant_pack, workspace.devPtr, cudnn_cuda_graph).is_good());

    // Put cudnn's cuda graph into main graph
    cudaGraphNode_t cudnn_node_in_main_graph;
    cudaGraphAddChildGraphNode(&cudnn_node_in_main_graph,
                               main_cuda_graph,
                               NULL,
                               0,
                               cudnn_cuda_graph);  // Note that this clones cudnn_cuda_graph.

    // It is safe to destroy cudnn_cuda_graph here.
    cudaGraphDestroy(cudnn_cuda_graph);

    //// Instantiate the main graph.
    cudaGraphExec_t cuda_graph_exec;
    cudaGraphInstantiate(&cuda_graph_exec, main_cuda_graph, NULL, NULL, 0);

    cudaGraphLaunch(cuda_graph_exec, 0);

    //// Functional correctness
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(
        cudaMemcpy(d_gpu.hostPtr, d_gpu.devPtr, sizeof(d_gpu.hostPtr[0]) * d_gpu.n_elems, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < d_gpu.n_elems; i++) {
        REQUIRE(__half2float(d_gpu.hostPtr[i]) ==
                scale_value * k * __half2float(starter_value) + __half2float(bias_value));
    }

    //// Update the instantiated cuda graph with new device pointers
    Surface<int8_t> workspace_new(graph->get_workspace_size(), false);

    half starter_value_new = __float2half(1.f);
    half bias_value_new    = __float2half(1.f);
    Surface<half> a_gpu_new(b * m * k, false, starter_value_new);
    Surface<half> b_gpu_new(b * k * n, false, starter_value_new);
    Surface<half> d_gpu_new(b * m * n, false);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void*> variant_pack_new = {
        {A_UID, a_gpu_new.devPtr}, {B_UID, b_gpu_new.devPtr}, {C_UID, &bias_value_new}, {D_UID, d_gpu_new.devPtr}};

    // This needs a cudnn cuda graph, which we can query from the cudnn_node in the main graph
    cudaGraph_t cudnn_cuda_graph_new;
    cudaGraphChildGraphNodeGetGraph(cudnn_node_in_main_graph, &cudnn_cuda_graph_new);

    REQUIRE(graph->update_cuda_graph(handle, variant_pack_new, workspace_new.devPtr, cudnn_cuda_graph_new).is_good());

    cudaGraphExecChildGraphNodeSetParams(cuda_graph_exec, cudnn_node_in_main_graph, cudnn_cuda_graph_new);

    cudaGraphLaunch(cuda_graph_exec, 0);

    //// Functional correctness
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(
        d_gpu_new.hostPtr, d_gpu_new.devPtr, sizeof(d_gpu_new.hostPtr[0]) * d_gpu_new.n_elems, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    for (int i = 0; i < d_gpu_new.n_elems; i++) {
        REQUIRE(__half2float(d_gpu_new.hostPtr[i]) ==
                (scale_value * k * __half2float(starter_value_new) + __half2float(bias_value_new)));
    }

    //// Cleanup
    cudaGraphExecDestroy(cuda_graph_exec);
    cudaGraphDestroy(main_cuda_graph);
    cudaGraphDestroy(cudnn_cuda_graph_new);
    cudnnDestroy(handle);
}