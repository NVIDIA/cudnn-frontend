/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CUDA graphs keep the execution plans and host data they were recorded from alive, and the
// frontend releases them again once those graphs are gone.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cudnn_frontend.h>

#define CUDA_CHECK(status)                                                                               \
    do {                                                                                                 \
        cudaError_t err_ = (status);                                                                     \
        if (err_ != cudaSuccess) {                                                                       \
            std::stringstream msg_;                                                                      \
            msg_ << "CUDA Error: " << cudaGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__; \
            FAIL(msg_.str());                                                                            \
        }                                                                                                \
    } while (0)

namespace {
struct CudnnHandleDeleter {
    void
    operator()(cudnnHandle_t *handle) const {
        if (handle != nullptr) {
            (void)cudnnDestroy(*handle);
            delete handle;
        }
    }
};

std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>
create_cudnn_handle() {
    auto handle = std::make_unique<cudnnHandle_t>();
    REQUIRE(cudnnCreate(handle.get()) == CUDNN_STATUS_SUCCESS);
    return std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>(handle.release());
}

// Device buffer, filled with one value or (for half) with uniform values in [0, 1).
template <typename T>
struct Surface {
    T *devPtr   = nullptr;
    size_t size = 0;

    explicit Surface(size_t n) : size(n) {
        std::vector<T> host(n, T{});
        if constexpr (std::is_same_v<T, half>) {
            unsigned seed = 123456789;
            for (auto &value : host) {
                seed  = 1103515245u * seed + 12345u;
                value = __float2half(static_cast<float>(seed) * 2.3283064e-10f);  // 2^-32
            }
        }
        upload(host);
    }
    Surface(size_t n, T value) : size(n) { upload(std::vector<T>(n, value)); }
    Surface(Surface const &) = delete;
    Surface &
    operator=(Surface const &) = delete;
    ~Surface() {
        if (devPtr != nullptr) {
            (void)cudaFree(devPtr);
        }
    }

   private:
    void
    upload(std::vector<T> const &host) {
        if (size == 0) {
            return;
        }
        CUDA_CHECK(cudaMalloc(&devPtr, size * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(devPtr, host.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
};
}  // namespace

#define A_UID 0
#define B_UID 1
#define C_UID 2
#define D_UID 3

namespace {
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

// Counts the payloads of CudaGraphRetainedResource that have been released.
int released_payloads = 0;

std::shared_ptr<void>
make_counted_payload() {
    return std::shared_ptr<void>(new int(0), [](void *p) {
        delete static_cast<int *>(p);
        ++released_payloads;
    });
}

// CUDA may run a user object's destructor asynchronously once its last reference is gone
// (cudaUserObjectNoDestructorSync), so a payload shows up in the deferred-release queue some time
// after its last CUDA graph is destroyed. Wait for that, for up to two seconds.
bool
wait_for_deferred_release() {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!cudnn_frontend::detail::CudaGraphRetainedResource::has_deferred_releases()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Queue one payload for release: an owner records it into a CUDA graph, the owner goes away, then
// the graph does.
void
queue_one_release() {
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    {
        cudnn_frontend::detail::CudaGraphRetainedResource owner;
        REQUIRE(owner.retain_on_graph(graph, make_counted_payload) == cudaSuccess);
    }
    CUDA_CHECK(cudaGraphDestroy(graph));
    REQUIRE(wait_for_deferred_release());
}
}  // namespace

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

    SECTION("recorded directly and retained by a cloned graphExec") {
        cudaGraph_t source;
        CUDA_CHECK(cudaGraphCreate(&source, 0));
        REQUIRE(graph->populate_cuda_graph_direct(handle, variant_pack, workspace.devPtr, source).is_good());
        cudaGraph_t clone;
        CUDA_CHECK(cudaGraphClone(&clone, source));
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaGraphInstantiate(&executable, clone, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(source));
        CUDA_CHECK(cudaGraphDestroy(clone));
        graph.reset();

        for (int replay = 0; replay < 3; ++replay) {
            churn();
            CUDA_CHECK(cudaGraphLaunch(executable, 0));
            verify_output();
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
    }

    SECTION("updated directly with an independent frontend plan") {
        cudaGraph_t source;
        CUDA_CHECK(cudaGraphCreate(&source, 0));
        REQUIRE(graph->populate_cuda_graph_direct(handle, variant_pack, workspace.devPtr, source).is_good());
        auto updater = create_graph(b, m, n, k, scale_value);
        REQUIRE(build(updater));
        Surface<half> updated_a(b * m * k, starter_value);
        auto updated_pack   = variant_pack;
        updated_pack[A_UID] = updated_a.devPtr;
        REQUIRE(updater->update_cuda_graph_direct(handle, updated_pack, workspace.devPtr, source).is_good());
        cudaGraph_t clone;
        CUDA_CHECK(cudaGraphClone(&clone, source));
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaGraphInstantiate(&executable, clone, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(source));
        CUDA_CHECK(cudaGraphDestroy(clone));
        graph.reset();
        updater.reset();

        for (int replay = 0; replay < 3; ++replay) {
            churn();
            CUDA_CHECK(cudaGraphLaunch(executable, 0));
            verify_output();
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
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
Only execution plans with runtime-compiled kernels are retained by the CUDA graphs recorded from
them. A plan of a precompiled engine (kernels that are part of the cuDNN library) is recorded
without a reference, and its CUDA graph keeps working after the frontend graph is gone.
*/
TEST_CASE("Precompiled plans need no retention by the CUDA graph", "[cudagraph][graph]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // A plain matmul, which precompiled engines serve.
    int64_t b = 1, m = 64, n = 64, k = 64;
    half starter_value   = __float2half(1.f);
    float const expected = static_cast<float>(k);

    Surface<half> a_gpu(b * m * k, starter_value);
    Surface<half> b_gpu(b * k * n, starter_value);
    Surface<half> d_gpu(b * m * n);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void *> variant_pack = {
        {A_UID, a_gpu.devPtr}, {B_UID, b_gpu.devPtr}, {D_UID, d_gpu.devPtr}};

    namespace fe = cudnn_frontend;
    auto graph   = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto A = graph->tensor(fe::graph::Tensor_attributes().set_uid(A_UID).set_dim({b, m, k}).set_stride({m * k, k, 1}));
    auto B = graph->tensor(fe::graph::Tensor_attributes().set_uid(B_UID).set_dim({b, k, n}).set_stride({k * n, n, 1}));
    auto D = graph->matmul(A, B, fe::graph::Matmul_attributes());
    D->set_output(true).set_uid(D_UID);

    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());
    graph->deselect_behavior_notes({fe::BehaviorNote_t::RUNTIME_COMPILATION});
    if (graph->check_support().is_bad()) {
        SKIP("No precompiled engine for this graph.");
    }
    REQUIRE(graph->build_plans().is_good());
    std::vector<fe::BehaviorNote_t> notes;
    REQUIRE(graph->get_behavior_notes(notes).is_good());
    REQUIRE(std::find(notes.begin(), notes.end(), fe::BehaviorNote_t::RUNTIME_COMPILATION) == notes.end());
    Surface<int8_t> workspace(graph->get_workspace_size());

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
        CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<half> d_host(d_gpu.size);
        CUDA_CHECK(cudaMemcpy(d_host.data(), d_gpu.devPtr, sizeof(half) * d_host.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < d_host.size(); i++) {
            REQUIRE(__half2float(d_host[i]) == expected);
        }
        CUDA_CHECK(cudaMemsetAsync(d_gpu.devPtr, 0xFF, sizeof(half) * d_gpu.size, stream));
    }

    CUDA_CHECK(cudaGraphExecDestroy(cuda_graph_exec));
    CUDA_CHECK(cudaGraphDestroy(captured_graph));
    REQUIRE(cudnnSetStream(handle, nullptr) == CUDNN_STATUS_SUCCESS);
    CUDA_CHECK(cudaStreamDestroy(stream));
#endif  // CUDART_VERSION < 12000
}

/*
Execution plans whose CUDA graphs are all gone are released lazily, when the
frontend next destroys an execution plan or graph. That release frees device
resources, which must not disturb a capture that is active at that moment on
ANOTHER stream in global mode (from this or any other thread).
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

    cudnn_frontend::detail::CudaGraphRetainedResource::drain_deferred_releases();  // left over from earlier tests
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

    // 1. Create everything the capture in step 3 needs first. While a global-mode capture is
    //    active, this thread may not allocate, free or synchronize, and building a graph
    //    destroys temporary plans, which would release the queued plan from step 2 early.
    auto captured = create_graph(b, m, n, k, scale_value);
    if (!build(captured)) {
        SKIP("No engine with the SUPPORTS_CUDA_GRAPH_NATIVE_API behavior note for this graph.");
    }
    Surface<int8_t> captured_workspace(captured->get_workspace_size());
    cudaStream_t stream_a;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking));

    // 2. Record a graph from a plan, then destroy everything: the plan's release is now queued.
    {
        auto doomed = create_graph(b, m, n, k, scale_value);
        REQUIRE(build(doomed));
        std::vector<cudnn_frontend::BehaviorNote_t> notes;
        REQUIRE(doomed->get_behavior_notes(notes).is_good());
        if (std::find(notes.begin(), notes.end(), cudnn_frontend::BehaviorNote_t::RUNTIME_COMPILATION) == notes.end()) {
            SKIP("The plan's engine is precompiled, so its CUDA graph does not retain it.");
        }
        Surface<int8_t> doomed_workspace(doomed->get_workspace_size());
        cudaGraph_t cudnn_cuda_graph;
        CUDA_CHECK(cudaGraphCreate(&cudnn_cuda_graph, 0));
        REQUIRE(doomed->populate_cuda_graph(handle, variant_pack, doomed_workspace.devPtr, cudnn_cuda_graph).is_good());
        doomed.reset();
        CUDA_CHECK(cudaGraphDestroy(cudnn_cuda_graph));
        REQUIRE(wait_for_deferred_release());
    }

    // 3. Start a global-mode capture on stream A.
    REQUIRE(cudnnSetStream(handle, stream_a) == CUDNN_STATUS_SUCCESS);
    CUDA_CHECK(cudaStreamBeginCapture(stream_a, cudaStreamCaptureModeGlobal));
    REQUIRE(captured->execute(handle, variant_pack, captured_workspace.devPtr).is_good());

    // 4. While the capture is active, release the queued plan from step 2. This is what
    //    destroying a frontend plan or graph does; it frees the plan's device resources and
    //    must not invalidate stream A's capture.
    REQUIRE(cudnn_frontend::detail::CudaGraphRetainedResource::has_deferred_releases());
    cudnn_frontend::detail::CudaGraphRetainedResource::drain_deferred_releases();
    REQUIRE_FALSE(cudnn_frontend::detail::CudaGraphRetainedResource::has_deferred_releases());

    // 5. Stream A's capture is still valid and replays correctly.
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
What a CUDA graph keeps alive is released only once no CUDA graph holds it, and then only when
the frontend destroys an owner (an execution plan, a graph's host-copy storage, ...). Never
during an execution: releasing frees memory or unloads code, which waits for all work on the
device, so an execute would block on, or deadlock with, unrelated streams.
*/
TEST_CASE("Retained resources are released when an owner is destroyed", "[cudagraph]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    using cudnn_frontend::detail::CudaGraphRetainedResource;
    CudaGraphRetainedResource::drain_deferred_releases();
    released_payloads = 0;

    SECTION("a graph that outlives its owner keeps the payload until the graph and an owner are gone") {
        cudaGraph_t graph;
        CUDA_CHECK(cudaGraphCreate(&graph, 0));
        {
            CudaGraphRetainedResource owner;
            REQUIRE(owner.retain_on_graph(graph, make_counted_payload) == cudaSuccess);
        }
        REQUIRE(released_payloads == 0);  // the graph still holds it

        auto other_owner = std::make_unique<CudaGraphRetainedResource>();
        CUDA_CHECK(cudaGraphDestroy(graph));
        REQUIRE(wait_for_deferred_release());
        REQUIRE(released_payloads == 0);

        // Recording work on a stream that is not being captured, as execute() does, releases nothing.
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        REQUIRE(other_owner->retain_on_capturing_stream(stream, make_counted_payload) == cudaSuccess);
        CUDA_CHECK(cudaStreamDestroy(stream));
        REQUIRE(released_payloads == 0);

        // Destroying any owner does.
        other_owner.reset();
        REQUIRE(released_payloads == 1);
        REQUIRE_FALSE(CudaGraphRetainedResource::has_deferred_releases());
    }

    SECTION("an owner that outlives its graphs releases the payload once it is destroyed") {
        cudaGraph_t graph;
        CUDA_CHECK(cudaGraphCreate(&graph, 0));
        {
            CudaGraphRetainedResource owner;
            REQUIRE(owner.retain_on_graph(graph, make_counted_payload) == cudaSuccess);
            CUDA_CHECK(cudaGraphDestroy(graph));
            REQUIRE(released_payloads == 0);  // the owner still holds it
        }
        // The owner held the last reference. If CUDA ran the user object's destructor while the owner
        // was destroyed, the owner released the payload itself; if it ran it asynchronously, the
        // payload is queued for the next owner destruction instead.
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (released_payloads == 0 && !CudaGraphRetainedResource::has_deferred_releases() &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CudaGraphRetainedResource::drain_deferred_releases();
        REQUIRE(released_payloads == 1);
        REQUIRE_FALSE(CudaGraphRetainedResource::has_deferred_releases());
    }
#endif
}

TEST_CASE("Executing a frontend graph does not release what CUDA graphs handed back", "[cudagraph][graph]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    using cudnn_frontend::detail::CudaGraphRetainedResource;
    CudaGraphRetainedResource::drain_deferred_releases();
    released_payloads = 0;

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t b = 8, m = 32, n = 16, k = 8;
    float scale_value  = .5f;
    half starter_value = __float2half(1.f);
    half bias_value    = __float2half(2.f);
    Surface<half> a_gpu(b * m * k, starter_value);
    Surface<half> b_gpu(b * k * n, starter_value);
    Surface<half> d_gpu(b * m * n);
    std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void *> variant_pack = {
        {A_UID, a_gpu.devPtr}, {B_UID, b_gpu.devPtr}, {C_UID, &bias_value}, {D_UID, d_gpu.devPtr}};

    // Built before anything is queued: building creates and drops plans, and dropping one releases.
    auto graph = create_graph(b, m, n, k, scale_value);
    REQUIRE(graph->validate().is_good());
    REQUIRE(graph->build_operation_graph(handle).is_good());
    REQUIRE(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good());
    REQUIRE(graph->check_support().is_good());
    REQUIRE(graph->build_plans().is_good());
    Surface<int8_t> workspace(graph->get_workspace_size());

    queue_one_release();
    REQUIRE(CudaGraphRetainedResource::has_deferred_releases());

    for (int i = 0; i < 3; ++i) {
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    REQUIRE(released_payloads == 0);
    REQUIRE(CudaGraphRetainedResource::has_deferred_releases());

    graph.reset();
    REQUIRE(released_payloads == 1);
    REQUIRE_FALSE(CudaGraphRetainedResource::has_deferred_releases());
#endif
}

/*
ALiBi slopes are computed on the host and copied into the workspace by a host-to-device
memcpy that is part of the recorded CUDA graph. That memcpy reads its host source on every
replay, so the source must live as long as the CUDA graph, even after the frontend graph that
computed it is destroyed.
*/
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

/*
With CudaGraphRetention_t::NONE, the CUDA graphs recorded from a frontend graph hold no reference to its
execution plans: destroying the frontend graph and then the CUDA graph releases nothing through the
deferred-release queue. With PER_GRAPH (the default), the CUDA graph's reference to the plan is queued
for release once the CUDA graph is gone. Both hold whether the plans come from heuristics or from
create_execution_plan() for a given engine.
*/
TEST_CASE("CUDA graph retention option", "[cudagraph][graph]") {
#if (CUDART_VERSION < 12000)
    SKIP("Test requires cuda toolkit 12.0 or above");
#else
    namespace fe = cudnn_frontend;
    using fe::detail::CudaGraphRetainedResource;
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    REQUIRE(cudnnSetStream(handle, stream) == CUDNN_STATUS_SUCCESS);

    int64_t const b = 1, h = 4, s = 64, d = 64;
    Surface<half> q_gpu(b * h * s * d);
    Surface<half> k_gpu(b * h * s * d);
    Surface<half> v_gpu(b * h * s * d);
    Surface<half> o_gpu(b * h * s * d);
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
        {101, q_gpu.devPtr}, {102, k_gpu.devPtr}, {103, v_gpu.devPtr}, {104, o_gpu.devPtr}};

    auto make_graph = [&]() {
        auto graph = std::make_shared<fe::graph::Graph>();
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
        auto [o, stats] = graph->sdpa(qkv("Q", 101),
                                      qkv("K", 102),
                                      qkv("V", 103),
                                      fe::graph::SDPA_attributes().set_generate_stats(false).set_attn_scale(0.125f));
        (void)stats;
        o->set_output(true).set_uid(104).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1});
        REQUIRE(graph->validate().is_good());
        REQUIRE(graph->build_operation_graph(handle).is_good());
        return graph;
    };

    // Builds the plan from heuristics, or with create_execution_plan() for the engine and knobs that
    // heuristics chose (manual_plan), then records it into a CUDA graph and destroys both graphs.
    auto record_and_release = [&](fe::CudaGraphRetention_t retention, bool manual_plan) -> bool {
        auto graph = make_graph();
        graph->set_cuda_graph_retention(retention);
        REQUIRE(graph->get_cuda_graph_retention() == retention);
        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
        graph->select_behavior_notes({fe::BehaviorNote_t::RUNTIME_COMPILATION});
        if (graph->check_support().is_bad()) {
            return false;
        }
        REQUIRE(graph->build_plans().is_good());
        if (manual_plan) {
            std::string chosen, name;
            REQUIRE(graph->get_plan_name(chosen).is_good());
            int64_t engine_id = -1;
            std::unordered_map<fe::KnobType_t, int64_t> knobs;
            for (int64_t i = 0; i < graph->get_execution_plan_count() && engine_id < 0; ++i) {
                if (graph->get_plan_name_at_index(i, name).is_good() && name == chosen) {
                    REQUIRE(graph->get_engine_and_knobs_at_index(i, engine_id, knobs).is_good());
                }
            }
            REQUIRE(engine_id >= 0);
            graph = make_graph();
            graph->set_cuda_graph_retention(retention);
            REQUIRE(graph->create_execution_plan(engine_id, knobs).is_good());
            REQUIRE(graph->build_plans().is_good());
        }
        Surface<int8_t> workspace(graph->get_workspace_size());

        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<half> reference(o_gpu.size);
        CUDA_CHECK(cudaMemcpy(reference.data(), o_gpu.devPtr, sizeof(half) * o_gpu.size, cudaMemcpyDeviceToHost));

        CudaGraphRetainedResource::drain_deferred_releases();  // start from an empty queue
        REQUIRE_FALSE(CudaGraphRetainedResource::has_deferred_releases());

        cudaGraph_t cuda_graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        CUDA_CHECK(cudaStreamEndCapture(stream, &cuda_graph));
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaGraphInstantiate(&exec, cuda_graph, nullptr, nullptr, 0));

        // The frontend graph stays alive across the replay, as NONE requires.
        CUDA_CHECK(cudaMemsetAsync(o_gpu.devPtr, 0, sizeof(half) * o_gpu.size, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<half> replayed(o_gpu.size);
        CUDA_CHECK(cudaMemcpy(replayed.data(), o_gpu.devPtr, sizeof(half) * o_gpu.size, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < replayed.size(); ++i) {
            REQUIRE(__half2float(replayed[i]) == __half2float(reference[i]));
        }

        graph.reset();
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(cuda_graph));
        return true;
    };

    // The setting reaches plans built from heuristics and plans built for a given engine alike.
    auto const manual_plan = GENERATE(false, true);
    CAPTURE(manual_plan);

    SECTION("PER_GRAPH: the CUDA graph's reference to the plan is released once the CUDA graph is gone") {
        if (!record_and_release(fe::CudaGraphRetention_t::PER_GRAPH, manual_plan)) {
            SKIP("No runtime-compiled engine for this SDPA graph.");
        }
        REQUIRE(wait_for_deferred_release());
        CudaGraphRetainedResource::drain_deferred_releases();
    }

    SECTION("NONE: the CUDA graph holds no reference") {
        if (!record_and_release(fe::CudaGraphRetention_t::NONE, manual_plan)) {
            SKIP("No runtime-compiled engine for this SDPA graph.");
        }
        // User-object destructors run asynchronously; give a release time to show up.
        for (int i = 0; i < 200 && !CudaGraphRetainedResource::has_deferred_releases(); ++i) {
            CUDA_CHECK(cudaDeviceSynchronize());
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE_FALSE(CudaGraphRetainedResource::has_deferred_releases());
    }

    REQUIRE(cudnnSetStream(handle, nullptr) == CUDNN_STATUS_SUCCESS);
    CUDA_CHECK(cudaStreamDestroy(stream));
#endif  // CUDART_VERSION < 12000
}
