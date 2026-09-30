/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils/helpers.h"
#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

/*
Run this example by using command:
bin/samples "Cuda graphs with matmul add"

This example shows how to construct a CUDA graph using cuDNN's
native CUDA graph API (as opposed to using CUDA graph capture),
using matmul add as the example operation.

In this example, the constructed CUDA graph is embedded as a
child of a larger CUDA graph (as we expect many users and
frameworks will want to do).

For a different example showing how to construct a CUDA graph and
execute it by itself, see ../sdpa/fp16_fwd_with_cudagraphs.cpp.
*/

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
    // cuDNN only supports native CUDA graphs in CUDA 12.0 and above.
    // Because the below test depends on some CUDA graph APIs that changed
    // between CUDA 11.x and 12.0, it wouldn't even compile in <12.0 anyway,
    // so we just disable the whole test by #if in that case.
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    // Also check the CUDA version at runtime, for good measure.
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    //// Main graph
    // This example shows how to add a cudnn cuda graph to an already existing
    // cuda graph.
    cudaGraph_t main_cuda_graph;
    cudaGraphCreate(&main_cuda_graph, 0);

    // Create any FE graph that you want to create a cuda graph for
    int64_t b = 8, m = 32, n = 16, k = 8;
    float scale_value = .5f;
    auto graph        = create_graph(b, m, n, k, scale_value);

    // Create the execution plan, as that is needed to populate cuda graph with
    // cudnn kernels
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // Validate the graph and lower the FE graph to BE graph
    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    REQUIRE(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());

    // Make sure the selected execution plan supports cuda graph
    graph->select_behavior_notes({cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
    auto status = graph->check_support();
    if (cudnn_frontend::detail::get_backend_version() >= 90500) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.is_bad());
        SKIP(
            "cudnn versions earlier than 9.5 don't support behavior note of "
            "SUPPORTS_CUDA_GRAPH_NATIVE_API.");
    }

    //// Test code
    // Does not necessarily need to be included in user code, in case you are
    // referring to this sample for your usecase.
    // START
    std::vector<cudnn_frontend::BehaviorNote_t> notes;
    status = graph->get_behavior_notes(notes);
    REQUIRE(status.is_bad());  // expected to fail as no candidate has been set yet

    notes.clear();
    status = graph->get_behavior_notes_for_plan_at_index(0, notes);
    REQUIRE(status.is_good());
    // Make sure that the note is SUPPORTS_CUDA_GRAPH
    bool supports_cuda_graph_native_api = false;
    for (auto note : notes) {
        if (note == cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API) {
            supports_cuda_graph_native_api = true;
        }
    }
    REQUIRE(supports_cuda_graph_native_api);
    // END

    REQUIRE(graph->build_plans().is_good());

    //// Test code
    // Does not necessarily need to be included in user code, in case you are
    // referring to this sample for your usecase.
    // START
    notes.clear();
    status = graph->get_behavior_notes(notes);
    REQUIRE(status.is_good());  // expected to pass now as candidate has been set

    // Make sure that the note is SUPPORTS_CUDA_GRAPH
    supports_cuda_graph_native_api = false;
    for (auto note : notes) {
        if (note == cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API) {
            supports_cuda_graph_native_api = true;
        }
    }
    REQUIRE(supports_cuda_graph_native_api);
    // END

    //// Populate an existing cuda graph with cudnn's cuda graph
    cudaGraph_t cudnn_cuda_graph;

    // Initialize the cudnn cuda graph.
    // The responsibility to destroy is on the user.
    cudaGraphCreate(&cudnn_cuda_graph, 0);  // 0 is just what the API says to pass

    Surface<int8_t> workspace(graph->get_workspace_size());

    half starter_value = __float2half(1.f);
    half bias_value    = __float2half(2.f);
    Surface<half> a_gpu(b * m * k, starter_value);
    Surface<half> b_gpu(b * k * n, starter_value);
    Surface<half> d_gpu(b * m * n);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void *> variant_pack = {
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
    std::vector<half> d_host(d_gpu.size);
    CUDA_CHECK(cudaMemcpy(d_host.data(), d_gpu.devPtr, sizeof(d_host[0]) * d_host.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < d_gpu.size; i++) {
        REQUIRE(__half2float(d_host[i]) == scale_value * k * __half2float(starter_value) + __half2float(bias_value));
    }

    //// Update the instantiated cuda graph with new device pointers
    Surface<int8_t> workspace_new(graph->get_workspace_size());

    half starter_value_new = __float2half(1.f);
    half bias_value_new    = __float2half(1.f);
    Surface<half> a_gpu_new(b * m * k, starter_value_new);
    Surface<half> b_gpu_new(b * k * n, starter_value_new);
    Surface<half> d_gpu_new(b * m * n);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void *> variant_pack_new = {
        {A_UID, a_gpu_new.devPtr}, {B_UID, b_gpu_new.devPtr}, {C_UID, &bias_value_new}, {D_UID, d_gpu_new.devPtr}};

    // This needs a cudnn cuda graph, which we can query from the cudnn_node in
    // the main graph
    cudaGraph_t cudnn_cuda_graph_new;
    cudaGraphChildGraphNodeGetGraph(cudnn_node_in_main_graph, &cudnn_cuda_graph_new);

    REQUIRE(graph->update_cuda_graph(handle, variant_pack_new, workspace_new.devPtr, cudnn_cuda_graph_new).is_good());

    cudaGraphExecChildGraphNodeSetParams(cuda_graph_exec, cudnn_node_in_main_graph, cudnn_cuda_graph_new);

    cudaGraphLaunch(cuda_graph_exec, 0);

    //// Functional correctness
    cudaDeviceSynchronize();
    std::vector<half> d_host_new(d_gpu_new.size);
    CUDA_CHECK(cudaMemcpy(
        d_host_new.data(), d_gpu_new.devPtr, sizeof(d_host_new[0]) * d_host_new.size(), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < d_gpu_new.size; i++) {
        REQUIRE(__half2float(d_host_new[i]) ==
                (scale_value * k * __half2float(starter_value_new) + __half2float(bias_value_new)));
    }

    //// Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
    CUDA_CHECK(cudaGraphDestroy(main_cuda_graph));
#endif  // CUDART_VERSION < 12000
}

/*
A CUDA graph recorded from a frontend graph must keep working after the
frontend graph, and with it the execution plan, is destroyed: the CUDA graph
keeps launching the plan's kernels, and cuDNN releases runtime-compiled kernel
code with the plan. The frontend gives the CUDA graph a
reference to the plan for exactly this reason. Both ways of recording are
covered: cuDNN's native CUDA graph API and CUDA stream capture.
*/
TEST_CASE("Cuda graph outlives the frontend graph", "[cudagraph][graph]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    if (cudnn_frontend::detail::get_backend_version() < 90500) {
        SKIP("cudnn versions earlier than 9.5 don't support the native CUDA graph API.");
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t b = 8, m = 32, n = 16, k = 8;
    float scale_value    = .5f;
    half starter_value   = __float2half(1.f);
    half bias_value      = __float2half(2.f);
    float const expected = scale_value * k * __half2float(starter_value) + __half2float(bias_value);

    Surface<half> a_gpu(b * m * k, starter_value);
    Surface<half> b_gpu(b * k * n, starter_value);
    Surface<half> d_gpu(b * m * n);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void *> variant_pack = {
        {A_UID, a_gpu.devPtr}, {B_UID, b_gpu.devPtr}, {C_UID, &bias_value}, {D_UID, d_gpu.devPtr}};

    // The native-API section needs an engine with SUPPORTS_CUDA_GRAPH_NATIVE_API; the capture
    // section uses the same plan so both sections exercise the same kernels.
    auto build = [&](std::shared_ptr<cudnn_frontend::graph::Graph> const &g) -> bool {
        REQUIRE(g->validate().is_good());
        REQUIRE(g->build_operation_graph(handle).is_good());
        REQUIRE(g->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());
        g->select_behavior_notes({cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
        if (g->check_support().is_bad()) {
            return false;
        }
        REQUIRE(g->build_plans().is_good());
        return true;
    };

    auto verify_output = [&]() {
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<half> d_host(d_gpu.size);
        CUDA_CHECK(cudaMemcpy(d_host.data(), d_gpu.devPtr, sizeof(d_host[0]) * d_host.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < d_host.size(); i++) {
            REQUIRE(__half2float(d_host[i]) == expected);
        }
        CUDA_CHECK(cudaMemset(d_gpu.devPtr, 0xFF, sizeof(half) * d_gpu.size));
    };

    // Build and run an unrelated plan, as an application keeps doing after recording a graph.
    // Without the plan retention, this is what recycles the destroyed plan's kernel code.
    auto churn = [&]() {
        auto donor = create_graph(b, 2 * m, n, k, scale_value);
        REQUIRE(build(donor));
        Surface<half> donor_a(b * 2 * m * k, starter_value);
        Surface<half> donor_d(b * 2 * m * n);
        Surface<int8_t> donor_workspace(donor->get_workspace_size());
        std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void *> donor_pack = {
            {A_UID, donor_a.devPtr}, {B_UID, b_gpu.devPtr}, {C_UID, &bias_value}, {D_UID, donor_d.devPtr}};
        REQUIRE(donor->execute(handle, donor_pack, donor_workspace.devPtr).is_good());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    auto graph = create_graph(b, m, n, k, scale_value);
    if (!build(graph)) {
        SKIP("No engine with the SUPPORTS_CUDA_GRAPH_NATIVE_API behavior note for this graph.");
    }
    // The workspace is baked into the recorded graph, so it must outlive every replay.
    Surface<int8_t> workspace(graph->get_workspace_size());

    SECTION("recorded with the native CUDA graph API") {
        cudaGraph_t main_cuda_graph;
        CUDA_CHECK(cudaGraphCreate(&main_cuda_graph, 0));
        cudaGraph_t cudnn_cuda_graph;
        CUDA_CHECK(cudaGraphCreate(&cudnn_cuda_graph, 0));
        REQUIRE(graph->populate_cuda_graph(handle, variant_pack, workspace.devPtr, cudnn_cuda_graph).is_good());
        cudaGraphNode_t cudnn_node;
        CUDA_CHECK(cudaGraphAddChildGraphNode(&cudnn_node, main_cuda_graph, nullptr, 0, cudnn_cuda_graph));
        CUDA_CHECK(cudaGraphDestroy(cudnn_cuda_graph));
        cudaGraphExec_t cuda_graph_exec;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec, main_cuda_graph, nullptr, nullptr, 0));

        // The application is done with the frontend graph; only the CUDA graph remains.
        graph.reset();

        for (int replay = 0; replay < 3; ++replay) {
            churn();
            CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, 0));
            verify_output();
        }

        CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
        CUDA_CHECK(cudaGraphDestroy(main_cuda_graph));
    }

    SECTION("recorded with stream capture") {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        REQUIRE(cudnnSetStream(handle, stream) == CUDNN_STATUS_SUCCESS);

        cudaGraph_t captured_graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured_graph));
        cudaGraphExec_t cuda_graph_exec;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec, captured_graph, nullptr, nullptr, 0));

        graph.reset();

        for (int replay = 0; replay < 3; ++replay) {
            churn();
            CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            verify_output();
        }

        CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
        CUDA_CHECK(cudaGraphDestroy(captured_graph));
        REQUIRE(cudnnSetStream(handle, nullptr) == CUDNN_STATUS_SUCCESS);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
#endif  // CUDART_VERSION < 12000
}

/*
Execution plans whose CUDA graphs are all gone are released lazily, on a later
execute() that is not being captured. That release frees device resources,
which must not disturb a capture that is active at that moment on ANOTHER
stream in global mode (from this or any other thread).
*/
TEST_CASE("Releasing retained plans does not disturb another stream's capture", "[cudagraph][graph]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    if (cudnn_frontend::detail::get_backend_version() < 90500) {
        SKIP("cudnn versions earlier than 9.5 don't support the native CUDA graph API.");
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t b = 8, m = 32, n = 16, k = 8;
    float scale_value    = .5f;
    half starter_value   = __float2half(1.f);
    half bias_value      = __float2half(2.f);
    float const expected = scale_value * k * __half2float(starter_value) + __half2float(bias_value);

    Surface<half> a_gpu(b * m * k, starter_value);
    Surface<half> b_gpu(b * k * n, starter_value);
    Surface<half> d_gpu(b * m * n);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void *> variant_pack = {
        {A_UID, a_gpu.devPtr}, {B_UID, b_gpu.devPtr}, {C_UID, &bias_value}, {D_UID, d_gpu.devPtr}};

    auto build = [&](std::shared_ptr<cudnn_frontend::graph::Graph> const &g) -> bool {
        REQUIRE(g->validate().is_good());
        REQUIRE(g->build_operation_graph(handle).is_good());
        REQUIRE(g->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());
        g->select_behavior_notes({cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
        if (g->check_support().is_bad()) {
            return false;
        }
        REQUIRE(g->build_plans().is_good());
        return true;
    };

    // 1. Record a graph from a plan, then destroy everything: the plan's release is now queued.
    {
        auto doomed = create_graph(b, m, n, k, scale_value);
        if (!build(doomed)) {
            SKIP("No engine with the SUPPORTS_CUDA_GRAPH_NATIVE_API behavior note for this graph.");
        }
        Surface<int8_t> doomed_workspace(doomed->get_workspace_size());
        cudaGraph_t cudnn_cuda_graph;
        CUDA_CHECK(cudaGraphCreate(&cudnn_cuda_graph, 0));
        REQUIRE(doomed->populate_cuda_graph(handle, variant_pack, doomed_workspace.devPtr, cudnn_cuda_graph).is_good());
        doomed.reset();
        CUDA_CHECK(cudaGraphDestroy(cudnn_cuda_graph));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 2. Start a global-mode capture on stream A. Everything the capture needs is created
    //    beforehand: while a global-mode capture is active, this thread may not allocate,
    //    free or synchronize.
    auto captured = create_graph(b, m, n, k, scale_value);
    REQUIRE(build(captured));
    Surface<int8_t> captured_workspace(captured->get_workspace_size());
    cudaStream_t stream_a;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking));
    REQUIRE(cudnnSetStream(handle, stream_a) == CUDNN_STATUS_SUCCESS);
    CUDA_CHECK(cudaStreamBeginCapture(stream_a, cudaStreamCaptureModeGlobal));
    REQUIRE(captured->execute(handle, variant_pack, captured_workspace.devPtr).is_good());

    // 3. While the capture is active, release the queued plan from step 1. This is what an
    //    execute() on a stream that is not capturing does; it frees the plan's device
    //    resources and must not invalidate stream A's capture.
    cudnn_frontend::detail::CudaGraphRetainedResource::drain_deferred_releases();

    // 4. Stream A's capture is still valid and replays correctly.
    cudaGraph_t captured_graph;
    CUDA_CHECK(cudaStreamEndCapture(stream_a, &captured_graph));
    cudaGraphExec_t cuda_graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec, captured_graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaMemset(d_gpu.devPtr, 0xFF, sizeof(half) * d_gpu.size));
    CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    std::vector<half> d_host(d_gpu.size);
    CUDA_CHECK(cudaMemcpy(d_host.data(), d_gpu.devPtr, sizeof(d_host[0]) * d_host.size(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < d_host.size(); i++) {
        REQUIRE(__half2float(d_host[i]) == expected);
    }

    CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
    CUDA_CHECK(cudaGraphDestroy(captured_graph));
    REQUIRE(cudnnSetStream(handle, nullptr) == CUDNN_STATUS_SUCCESS);
    CUDA_CHECK(cudaStreamDestroy(stream_a));
#endif  // CUDART_VERSION < 12000
}

/*
ALiBi slopes are computed on the host and copied into the workspace by a host-to-device
memcpy that is part of the recorded CUDA graph. That memcpy reads its host source on every
replay, so the source must live as long as the CUDA graph, even after the frontend graph that
computed it is destroyed.
*/
namespace {
std::shared_ptr<cudnn_frontend::graph::Graph>
create_alibi_sdpa_graph(int64_t b, int64_t h, int64_t s, int64_t d) {
    namespace fe = cudnn_frontend;
    auto graph   = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto qkv = [&](char const *name, int64_t uid) {
        return graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name(name)
                                 .set_uid(uid)
                                 .set_dim({b, h, s, d})
                                 .set_stride({h * s * d, s * d, d, 1}));
    };
    auto q       = qkv("Q", 101);
    auto k       = qkv("K", 102);
    auto v       = qkv("V", 103);
    auto options = fe::graph::SDPA_attributes()
                       .set_name("sdpa_alibi")
                       .set_generate_stats(false)
                       .set_alibi_mask(true)
                       .set_attn_scale(0.125f)
                       .set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                       .set_diagonal_band_right_bound(0);
    auto [o, stats] = graph->sdpa(q, k, v, options);
    (void)stats;
    o->set_output(true).set_uid(104).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1});
    return graph;
}
}  // namespace

TEST_CASE("ALiBi slopes outlive the frontend graph in a CUDA graph", "[cudagraph][graph]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const b = 2, h = 8, s = 128, d = 64;
    Surface<half> q_gpu(b * h * s * d);
    Surface<half> k_gpu(b * h * s * d);
    Surface<half> v_gpu(b * h * s * d);
    Surface<half> o_gpu(b * h * s * d);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void *> variant_pack = {
        {101, q_gpu.devPtr}, {102, k_gpu.devPtr}, {103, v_gpu.devPtr}, {104, o_gpu.devPtr}};

    auto build = [&](std::shared_ptr<cudnn_frontend::graph::Graph> const &g, bool native_api) -> bool {
        if (g->validate().is_bad() || g->build_operation_graph(handle).is_bad()) {
            return false;
        }
        REQUIRE(g->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());
        if (native_api) {
            g->select_behavior_notes({cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
        }
        if (g->check_support().is_bad()) {
            return false;
        }
        REQUIRE(g->build_plans().is_good());
        return true;
    };

    auto read_output = [&](Surface<half> &out) {
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<half> host(out.size);
        CUDA_CHECK(cudaMemcpy(host.data(), out.devPtr, sizeof(half) * host.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(out.devPtr, 0, sizeof(half) * out.size));
        return host;
    };
    auto read_o = [&]() { return read_output(o_gpu); };

    // Allocate and fill host blocks of the ALiBi slopes' size, as the application's own work does
    // after recording the graph: this is what reuses a slopes buffer that was freed too early.
    auto churn_host_heap = [&]() {
        std::vector<std::vector<float> > junk;
        for (int i = 0; i < 4096; ++i) {
            junk.emplace_back(static_cast<size_t>(h), 1.0e4f);
        }
        return junk;
    };

    auto expect_replays_match = [&](std::vector<half> const &reference, auto &&replay, Surface<half> *out = nullptr) {
        for (int round = 0; round < 3; ++round) {
            auto junk = churn_host_heap();
            replay();
            auto replayed     = read_output(out != nullptr ? *out : o_gpu);
            size_t mismatches = 0;
            for (size_t i = 0; i < replayed.size(); ++i) {
                mismatches += (__half2float(replayed[i]) != __half2float(reference[i])) ? 1 : 0;
            }
            INFO("replay round " << round);
            REQUIRE(mismatches == 0);
        }
    };

    SECTION("recorded with stream capture") {
        auto graph = create_alibi_sdpa_graph(b, h, s, d);
        if (!build(graph, /*native_api=*/false)) {
            SKIP("SDPA with an ALiBi mask is not supported on this configuration.");
        }
        Surface<int8_t> workspace(graph->get_workspace_size());

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        REQUIRE(cudnnSetStream(handle, stream) == CUDNN_STATUS_SUCCESS);

        // Eager reference from the same plan.
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        auto reference = read_o();

        cudaGraph_t captured_graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured_graph));
        cudaGraphExec_t cuda_graph_exec;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec, captured_graph, nullptr, nullptr, 0));

        graph.reset();
        expect_replays_match(reference, [&]() {
            CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });

        CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
        CUDA_CHECK(cudaGraphDestroy(captured_graph));
        REQUIRE(cudnnSetStream(handle, nullptr) == CUDNN_STATUS_SUCCESS);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    SECTION("recorded with the native CUDA graph API") {
        if (cudnn_frontend::detail::get_backend_version() < 90500) {
            SKIP("cudnn versions earlier than 9.5 don't support the native CUDA graph API.");
        }
        auto graph = create_alibi_sdpa_graph(b, h, s, d);
        if (!build(graph, /*native_api=*/true)) {
            SKIP("No engine with the SUPPORTS_CUDA_GRAPH_NATIVE_API behavior note for SDPA with ALiBi.");
        }
        Surface<int8_t> workspace(graph->get_workspace_size());

        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        auto reference = read_o();

        cudaGraph_t cudnn_cuda_graph;
        CUDA_CHECK(cudaGraphCreate(&cudnn_cuda_graph, 0));
        REQUIRE(graph->populate_cuda_graph(handle, variant_pack, workspace.devPtr, cudnn_cuda_graph).is_good());
        cudaGraphExec_t cuda_graph_exec;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec, cudnn_cuda_graph, nullptr, nullptr, 0));

        graph.reset();
        expect_replays_match(reference, [&]() {
            CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, 0));
            CUDA_CHECK(cudaDeviceSynchronize());
        });

        CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
        CUDA_CHECK(cudaGraphDestroy(cudnn_cuda_graph));
    }

    SECTION("updated with the native CUDA graph API") {
        if (cudnn_frontend::detail::get_backend_version() < 90500) {
            SKIP("cudnn versions earlier than 9.5 don't support the native CUDA graph API.");
        }
        auto populating_graph = create_alibi_sdpa_graph(b, h, s, d);
        if (!build(populating_graph, /*native_api=*/true)) {
            SKIP("No engine with the SUPPORTS_CUDA_GRAPH_NATIVE_API behavior note for SDPA with ALiBi.");
        }
        Surface<int8_t> workspace(populating_graph->get_workspace_size());

        REQUIRE(populating_graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        auto reference = read_o();

        cudaGraph_t cudnn_cuda_graph;
        CUDA_CHECK(cudaGraphCreate(&cudnn_cuda_graph, 0));
        REQUIRE(
            populating_graph->populate_cuda_graph(handle, variant_pack, workspace.devPtr, cudnn_cuda_graph).is_good());
        cudaGraphExec_t cuda_graph_exec;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec, cudnn_cuda_graph, nullptr, nullptr, 0));

        // A separately built frontend graph (e.g. a plan re-created after a cache miss) updates the
        // CUDA graph to write a different output. The memcpy node now reads the ALiBi slopes owned by
        // this second graph, which is destroyed right after the update.
        auto updating_graph = create_alibi_sdpa_graph(b, h, s, d);
        REQUIRE(build(updating_graph, /*native_api=*/true));
        REQUIRE(updating_graph->get_workspace_size() <= populating_graph->get_workspace_size());
        Surface<half> o2_gpu(b * h * s * d);
        auto updated_pack = variant_pack;
        updated_pack[104] = o2_gpu.devPtr;
        REQUIRE(updating_graph->update_cuda_graph(handle, updated_pack, workspace.devPtr, cudnn_cuda_graph).is_good());
        cudaGraphExecUpdateResultInfo update_info;
        CUDA_CHECK(cudaGraphExecUpdate(cuda_graph_exec, cudnn_cuda_graph, &update_info));

        populating_graph.reset();
        updating_graph.reset();
        expect_replays_match(
            reference,
            [&]() {
                CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, 0));
                CUDA_CHECK(cudaDeviceSynchronize());
            },
            &o2_gpu);

        CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
        CUDA_CHECK(cudaGraphDestroy(cudnn_cuda_graph));
    }
#endif  // CUDART_VERSION < 12000
}
