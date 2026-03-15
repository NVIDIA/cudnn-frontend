/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cudnn_frontend.h>
#include <cudnn_frontend/experimental/sm90_sdpa_prefill_engine.h>
#include <cudnn_frontend/experimental/sm100_sdpa_prefill_engine.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace fe = cudnn_frontend;

// UID constants for the cuDNN graph variant pack
#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define MAX_UID 6
#define SUM_EXP_UID 7

// ---------------------------------------------------------------------------
// Helper: returns true if the current GPU is Hopper (SM90) or Blackwell
// (SM100), i.e. any architecture supported by the OSS prefill engines.
// ---------------------------------------------------------------------------
static bool
is_oss_supported_arch() {
    return is_hopper_arch() || is_blackwell_computing_arch();
}

// ---------------------------------------------------------------------------
// Helper: build an SDPA forward graph that produces O, max, and sum_exp.
//
// Tensor layout is BSHD (batch, seqlen, heads, dim) so that the same device
// memory can be consumed by both the cuDNN graph path and the NVRTC engine
// path without a layout transposition.
// ---------------------------------------------------------------------------
static std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph(int64_t b,
                          int64_t h_q,
                          int64_t h_kv,
                          int64_t s_q,
                          int64_t s_kv,
                          int64_t d,
                          float attn_scale) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // BSHD-contiguous strides: for dim ordering (b, h, s, d), the physical
    // layout is (B, S, H, D) with:
    //   stride_d = 1
    //   stride_h = d
    //   stride_s = h * d
    //   stride_b = s * h * d
    // The graph API dims are (b, h, s, d), so we specify strides that
    // correspond to BSHD physical ordering.

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim({b, h_q, s_q, d})
                               .set_stride({h_q * s_q * d,  // stride for b
                                            d,              // stride for h
                                            h_q * d,        // stride for s
                                            1}));           // stride for d

    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim({b, h_kv, s_kv, d})
                               .set_stride({h_kv * s_kv * d, d, h_kv * d, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim({b, h_kv, s_kv, d})
                               .set_stride({h_kv * s_kv * d, d, h_kv * d, 1}));

    auto sdpa_options =
        fe::graph::SDPA_attributes().set_name("flash_attention").set_generate_stats(false).set_attn_scale(attn_scale);

    // Enable causal mask (top-left aligned, right bound = 0)
    sdpa_options.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT).set_diagonal_band_right_bound(0);

    // Request max and sum_exp auxiliary outputs
    auto Max = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("Max")
                                 .set_uid(MAX_UID)
                                 .set_dim({b, h_q, s_q, 1})
                                 .set_stride({h_q * s_q, s_q, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    sdpa_options.set_logit_max(Max);

    auto Sum_exp = graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Sum_exp")
                                     .set_uid(SUM_EXP_UID)
                                     .set_dim({b, h_q, s_q, 1})
                                     .set_stride({h_q * s_q, s_q, 1, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));
    sdpa_options.set_score_sum_exp(Sum_exp);

    auto [O, Stats] = graph->sdpa(Q, K, V, std::move(sdpa_options));

    // Output O uses the same BSHD layout
    O->set_output(true)
        .set_dim({b, h_q, s_q, d})
        .set_stride({h_q * s_q * d,  // stride for b
                     d,              // stride for h
                     h_q * d,        // stride for s
                     1})             // stride for d
        .set_uid(O_UID);

    // Stats is nullptr because generate_stats=false
    (void)Stats;

    return graph;
}

// ---------------------------------------------------------------------------
// BHSD layout: dims (B,H,S,D) with BHSD-contiguous strides {H*S*D, S*D, D, 1}
// This is the PyTorch-native contiguous layout.
// ---------------------------------------------------------------------------
static std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph_bhsd(int64_t b,
                               int64_t h_q,
                               int64_t h_kv,
                               int64_t s_q,
                               int64_t s_kv,
                               int64_t d,
                               float attn_scale) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // BHSD-contiguous strides: physical ordering is (B, H, S, D)
    //   stride_d = 1, stride_s = d, stride_h = s*d, stride_b = h*s*d
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim({b, h_q, s_q, d})
                               .set_stride({h_q * s_q * d, s_q * d, d, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim({b, h_kv, s_kv, d})
                               .set_stride({h_kv * s_kv * d, s_kv * d, d, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim({b, h_kv, s_kv, d})
                               .set_stride({h_kv * s_kv * d, s_kv * d, d, 1}));

    auto sdpa_options =
        fe::graph::SDPA_attributes().set_name("flash_attention").set_generate_stats(false).set_attn_scale(attn_scale);
    sdpa_options.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT).set_diagonal_band_right_bound(0);

    auto Max = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("Max")
                                 .set_uid(MAX_UID)
                                 .set_dim({b, h_q, s_q, 1})
                                 .set_stride({h_q * s_q, s_q, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    sdpa_options.set_logit_max(Max);

    auto Sum_exp = graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Sum_exp")
                                     .set_uid(SUM_EXP_UID)
                                     .set_dim({b, h_q, s_q, 1})
                                     .set_stride({h_q * s_q, s_q, 1, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));
    sdpa_options.set_score_sum_exp(Sum_exp);

    auto [O, Stats] = graph->sdpa(Q, K, V, std::move(sdpa_options));
    O->set_output(true).set_dim({b, h_q, s_q, d}).set_stride({h_q * s_q * d, s_q * d, d, 1}).set_uid(O_UID);
    (void)Stats;
    return graph;
}

// ---------------------------------------------------------------------------
// Helper: element-wise comparison of two half-precision buffers on host
// ---------------------------------------------------------------------------
static void
compare_half_outputs(void *gpu_a, void *gpu_b, int64_t num_elems, double atol, double rtol, const char *label) {
    std::vector<half> h_a(num_elems);
    std::vector<half> h_b(num_elems);
    CUDA_CHECK(cudaMemcpy(h_a.data(), gpu_a, num_elems * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b.data(), gpu_b, num_elems * sizeof(half), cudaMemcpyDeviceToHost));

    int64_t num_mismatches = 0;
    double max_abs_diff    = 0.0;
    double sum_abs_diff    = 0.0;

    for (int64_t i = 0; i < num_elems; ++i) {
        float va = cpu_half2float(h_a[i]);
        float vb = cpu_half2float(h_b[i]);
        double d = std::fabs(static_cast<double>(va) - static_cast<double>(vb));
        sum_abs_diff += d;
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > atol + rtol * std::fabs(static_cast<double>(va))) num_mismatches++;
    }

    double mean_abs_diff = sum_abs_diff / static_cast<double>(num_elems);
    std::printf("\n===== %s =====\n", label);
    std::printf("  Elements: %ld  Max diff: %.6e  Mean diff: %.6e  Mismatches: %ld\n",
                static_cast<long>(num_elems),
                max_abs_diff,
                mean_abs_diff,
                static_cast<long>(num_mismatches));
    std::printf("==========================================\n\n");
    REQUIRE(num_mismatches == 0);
}

// ---------------------------------------------------------------------------
// BSHD stride helper: physical layout (B, S, H, D) expressed in graph API
// dim order (B, H, S, D).
// ---------------------------------------------------------------------------
static std::vector<int64_t>
bshd_strides(int64_t h, int64_t s, int64_t d) {
    return {h * s * d, d, h * d, 1};
}

// ---------------------------------------------------------------------------
// TEST_CASE: Compare cuDNN graph SDPA output against NVRTC engine output.
// Uses HeurMode_t::OPENSOURCE which auto-dispatches to the correct engine
// for the current architecture (SM90 or SM100).
// ---------------------------------------------------------------------------
TEST_CASE("Prefill OSS Engine vs cuDNN Graph", "[graph][sdpa][engine][oss]") {
    // ---- Fixed problem configuration ----
    int64_t const b    = 2;
    int64_t const h_q  = 4;
    int64_t const h_kv = 2;
    int64_t const s_q  = 1024;
    int64_t const s_kv = 2048;
    int64_t const d    = 128;

    float const attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

    // ---- Skip if not a supported architecture ----
    if (!is_oss_supported_arch()) {
        SKIP("Test requires Hopper (SM90) or Blackwell (SM100) architecture");
        return;
    }

    // ---- cuDNN handle ----
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // ---- Shared input tensors (BSHD layout, half precision) ----
    int64_t const q_elems   = b * h_q * s_q * d;
    int64_t const k_elems   = b * h_kv * s_kv * d;
    int64_t const v_elems   = b * h_kv * s_kv * d;
    int64_t const o_elems   = b * h_q * s_q * d;
    int64_t const aux_elems = b * h_q * s_q;  // for max and sum_exp

    Surface<half> q_tensor(q_elems, false);
    Surface<half> k_tensor(k_elems, false);
    Surface<half> v_tensor(v_elems, false);

    // ================================================================
    // Path 1: cuDNN Graph API (reference)
    // ================================================================
    Surface<half> o_cudnn(o_elems, false);
    Surface<float> max_cudnn(aux_elems, false);
    Surface<float> sum_exp_cudnn(aux_elems, false);

    {
        auto graph = create_sdpa_forward_graph(b, h_q, h_kv, s_q, s_kv, d, attn_scale);

        auto status = graph->validate();
        REQUIRE(status.is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());
        auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
        REQUIRE(graph->check_support(handle).is_good());
        REQUIRE(graph->build_plans(handle).is_good());

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
            {Q_UID, q_tensor.devPtr},
            {K_UID, k_tensor.devPtr},
            {V_UID, v_tensor.devPtr},
            {O_UID, o_cudnn.devPtr},
            {MAX_UID, max_cudnn.devPtr},
            {SUM_EXP_UID, sum_exp_cudnn.devPtr},
        };

        int64_t workspace_size = 0;
        REQUIRE(graph->get_workspace_size(workspace_size).is_good());
        Surface<int8_t> workspace(workspace_size, false);

        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- Benchmark cuDNN graph path ----
        int const warmup_iters = 10;
        int const bench_iters  = 100;

        for (int i = 0; i < warmup_iters; ++i) {
            REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start_cudnn, stop_cudnn;
        CUDA_CHECK(cudaEventCreate(&start_cudnn));
        CUDA_CHECK(cudaEventCreate(&stop_cudnn));

        CUDA_CHECK(cudaEventRecord(start_cudnn));
        for (int i = 0; i < bench_iters; ++i) {
            REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        }
        CUDA_CHECK(cudaEventRecord(stop_cudnn));
        CUDA_CHECK(cudaEventSynchronize(stop_cudnn));

        float cudnn_total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&cudnn_total_ms, start_cudnn, stop_cudnn));
        float cudnn_mean_ms = cudnn_total_ms / static_cast<float>(bench_iters);

        CUDA_CHECK(cudaEventDestroy(start_cudnn));
        CUDA_CHECK(cudaEventDestroy(stop_cudnn));

        std::printf("\n[cuDNN Graph]  Mean kernel time: %.4f ms  (over %d iters, %d warmup)\n",
                    cudnn_mean_ms,
                    bench_iters,
                    warmup_iters);
    }

    // ================================================================
    // Path 2: Same graph structure, but using OPENSOURCE mode
    // (no manual engine instantiation -- everything goes through graph API)
    // ================================================================
    Surface<half> o_engine(o_elems, false);
    Surface<float> max_engine(aux_elems, false);
    Surface<float> sum_exp_engine(aux_elems, false);

    {
        auto oss_graph = create_sdpa_forward_graph(b, h_q, h_kv, s_q, s_kv, d, attn_scale);

        auto status = oss_graph->validate();
        REQUIRE(status.is_good());

        REQUIRE(oss_graph->build_operation_graph(handle).is_good());
        auto plans_status = oss_graph->create_execution_plans({fe::HeurMode_t::OPENSOURCE});
        REQUIRE(plans_status.is_good());
        REQUIRE(oss_graph->check_support(handle).is_good());
        REQUIRE(oss_graph->build_plans(handle).is_good());

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> oss_variant_pack = {
            {Q_UID, q_tensor.devPtr},
            {K_UID, k_tensor.devPtr},
            {V_UID, v_tensor.devPtr},
            {O_UID, o_engine.devPtr},
            {MAX_UID, max_engine.devPtr},
            {SUM_EXP_UID, sum_exp_engine.devPtr},
        };

        int64_t oss_workspace_size = 0;
        REQUIRE(oss_graph->get_workspace_size(oss_workspace_size).is_good());
        Surface<int8_t> oss_workspace(oss_workspace_size, false);

        auto exec_status = oss_graph->execute(handle, oss_variant_pack, oss_workspace.devPtr);
        if (exec_status.is_bad()) {
            std::printf("OSS kernel execute error: %s\n", exec_status.get_message().c_str());
        }
        REQUIRE(exec_status.is_good());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- Benchmark OSS engine path ----
        int const warmup_iters = 10;
        int const bench_iters  = 100;

        for (int i = 0; i < warmup_iters; ++i) {
            REQUIRE(oss_graph->execute(handle, oss_variant_pack, oss_workspace.devPtr).is_good());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start_oss, stop_oss;
        CUDA_CHECK(cudaEventCreate(&start_oss));
        CUDA_CHECK(cudaEventCreate(&stop_oss));

        CUDA_CHECK(cudaEventRecord(start_oss));
        for (int i = 0; i < bench_iters; ++i) {
            REQUIRE(oss_graph->execute(handle, oss_variant_pack, oss_workspace.devPtr).is_good());
        }
        CUDA_CHECK(cudaEventRecord(stop_oss));
        CUDA_CHECK(cudaEventSynchronize(stop_oss));

        float oss_total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&oss_total_ms, start_oss, stop_oss));
        float oss_mean_ms = oss_total_ms / static_cast<float>(bench_iters);

        CUDA_CHECK(cudaEventDestroy(start_oss));
        CUDA_CHECK(cudaEventDestroy(stop_oss));

        std::printf("[OSS Engine]   Mean kernel time: %.4f ms  (over %d iters, %d warmup)\n",
                    oss_mean_ms,
                    bench_iters,
                    warmup_iters);
    }

    // ================================================================
    // Comparison: download O tensors and compare element-wise
    // ================================================================
    {
        // Download cuDNN graph output to host
        std::vector<half> h_o_cudnn(o_elems);
        CUDA_CHECK(cudaMemcpy(h_o_cudnn.data(), o_cudnn.devPtr, o_elems * sizeof(half), cudaMemcpyDeviceToHost));

        // Download engine output to host
        std::vector<half> h_o_engine(o_elems);
        CUDA_CHECK(cudaMemcpy(h_o_engine.data(), o_engine.devPtr, o_elems * sizeof(half), cudaMemcpyDeviceToHost));

        // Element-wise comparison with tolerance
        double const atol = 5e-2;  // absolute tolerance (half-precision limited)
        double const rtol = 5e-2;  // relative tolerance

        int64_t num_mismatches = 0;
        double max_abs_diff    = 0.0;
        double sum_abs_diff    = 0.0;

        for (int64_t i = 0; i < o_elems; ++i) {
            float val_cudnn  = cpu_half2float(h_o_cudnn[i]);
            float val_engine = cpu_half2float(h_o_engine[i]);
            double abs_diff  = std::fabs(static_cast<double>(val_cudnn) - static_cast<double>(val_engine));
            double threshold = atol + rtol * std::fabs(static_cast<double>(val_cudnn));

            sum_abs_diff += abs_diff;
            if (abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
            }
            if (abs_diff > threshold) {
                num_mismatches++;
            }
        }

        double mean_abs_diff = sum_abs_diff / static_cast<double>(o_elems);

        // Print comparison statistics
        std::printf("\n===== Prefill Engine vs cuDNN Graph Comparison =====\n");
        std::printf("  Total elements:  %ld\n", static_cast<long>(o_elems));
        std::printf("  Max abs diff:    %.6e\n", max_abs_diff);
        std::printf("  Mean abs diff:   %.6e\n", mean_abs_diff);
        std::printf("  Mismatches:      %ld / %ld (atol=%.1e, rtol=%.1e)\n",
                    static_cast<long>(num_mismatches),
                    static_cast<long>(o_elems),
                    atol,
                    rtol);
        std::printf("====================================================\n\n");

        REQUIRE(num_mismatches == 0);
    }
}

// ---------------------------------------------------------------------------
// TEST_CASE: BHSD layout -- dims (B,H,S,D) with BHSD-contiguous strides.
// Uses HeurMode_t::OPENSOURCE which auto-dispatches to the correct engine.
// ---------------------------------------------------------------------------
TEST_CASE("Prefill OSS Engine BHSD layout", "[graph][sdpa][oss][bhsd]") {
    int64_t const b        = 2;
    int64_t const h_q      = 4;
    int64_t const h_kv     = 2;
    int64_t const s_q      = 1024;
    int64_t const s_kv     = 2048;
    int64_t const d        = 64;
    float const attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

    if (!is_oss_supported_arch()) {
        SKIP("Test requires Hopper (SM90) or Blackwell (SM100) architecture");
        return;
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    int64_t const q_elems   = b * h_q * s_q * d;
    int64_t const k_elems   = b * h_kv * s_kv * d;
    int64_t const v_elems   = b * h_kv * s_kv * d;
    int64_t const o_elems   = b * h_q * s_q * d;
    int64_t const aux_elems = b * h_q * s_q;

    Surface<half> q_tensor(q_elems, false);
    Surface<half> k_tensor(k_elems, false);
    Surface<half> v_tensor(v_elems, false);

    // ---- Path 1: cuDNN reference (BHSD layout) ----
    Surface<half> o_cudnn(o_elems, false);
    Surface<float> max_cudnn(aux_elems, false);
    Surface<float> sum_exp_cudnn(aux_elems, false);
    {
        auto graph = create_sdpa_forward_graph_bhsd(b, h_q, h_kv, s_q, s_kv, d, attn_scale);
        REQUIRE(graph->validate().is_good());
        REQUIRE(graph->build_operation_graph(handle).is_good());
        auto plans_status = graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        REQUIRE(plans_status.is_good());
        REQUIRE(graph->check_support(handle).is_good());
        REQUIRE(graph->build_plans(handle).is_good());

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
            {Q_UID, q_tensor.devPtr},
            {K_UID, k_tensor.devPtr},
            {V_UID, v_tensor.devPtr},
            {O_UID, o_cudnn.devPtr},
            {MAX_UID, max_cudnn.devPtr},
            {SUM_EXP_UID, sum_exp_cudnn.devPtr},
        };
        int64_t ws = 0;
        REQUIRE(graph->get_workspace_size(ws).is_good());
        Surface<int8_t> workspace(ws, false);
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Path 2: OPENSOURCE engine (BHSD layout) ----
    Surface<half> o_oss(o_elems, false);
    Surface<float> max_oss(aux_elems, false);
    Surface<float> sum_exp_oss(aux_elems, false);
    {
        auto graph = create_sdpa_forward_graph_bhsd(b, h_q, h_kv, s_q, s_kv, d, attn_scale);
        REQUIRE(graph->validate().is_good());
        REQUIRE(graph->build_operation_graph(handle).is_good());
        auto plans_status = graph->create_execution_plans({fe::HeurMode_t::OPENSOURCE});
        REQUIRE(plans_status.is_good());
        REQUIRE(graph->check_support(handle).is_good());
        REQUIRE(graph->build_plans(handle).is_good());

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack = {
            {Q_UID, q_tensor.devPtr},
            {K_UID, k_tensor.devPtr},
            {V_UID, v_tensor.devPtr},
            {O_UID, o_oss.devPtr},
            {MAX_UID, max_oss.devPtr},
            {SUM_EXP_UID, sum_exp_oss.devPtr},
        };
        int64_t ws = 0;
        REQUIRE(graph->get_workspace_size(ws).is_good());
        Surface<int8_t> workspace(ws, false);
        auto exec_status = graph->execute(handle, variant_pack, workspace.devPtr);
        if (exec_status.is_bad()) {
            std::printf("OSS execute error (BHSD): %s\n", exec_status.get_message().c_str());
        }
        REQUIRE(exec_status.is_good());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Compare ----
    {
        std::vector<half> h_o_cudnn(o_elems);
        CUDA_CHECK(cudaMemcpy(h_o_cudnn.data(), o_cudnn.devPtr, o_elems * sizeof(half), cudaMemcpyDeviceToHost));
        std::vector<half> h_o_oss(o_elems);
        CUDA_CHECK(cudaMemcpy(h_o_oss.data(), o_oss.devPtr, o_elems * sizeof(half), cudaMemcpyDeviceToHost));

        double const atol      = 5e-2;
        double const rtol      = 5e-2;
        int64_t num_mismatches = 0;
        double max_abs_diff    = 0.0;

        for (int64_t i = 0; i < o_elems; ++i) {
            float a     = cpu_half2float(h_o_cudnn[i]);
            float b_val = cpu_half2float(h_o_oss[i]);
            double diff = std::fabs(static_cast<double>(a) - static_cast<double>(b_val));
            if (diff > max_abs_diff) max_abs_diff = diff;
            if (diff > atol + rtol * std::fabs(static_cast<double>(a))) num_mismatches++;
        }

        std::printf("\n===== Prefill OSS Engine BHSD Layout =====\n");
        std::printf("  Elements: %ld  Max diff: %.6e  Mismatches: %ld\n",
                    static_cast<long>(o_elems),
                    max_abs_diff,
                    static_cast<long>(num_mismatches));
        std::printf("==========================================\n\n");
        REQUIRE(num_mismatches == 0);
    }
}

// ---------------------------------------------------------------------------
// TEST_CASE: Dynamic shapes -- build OSS graph once, execute with two
// different shape configurations, compare each against a separate cuDNN
// reference graph. Uses HeurMode_t::OPENSOURCE which auto-dispatches.
// ---------------------------------------------------------------------------
TEST_CASE("Prefill OSS Engine Dynamic Shapes", "[graph][sdpa][oss][dynamic]") {
    // ---- Initial shape configuration ----
    int64_t const b    = 2;
    int64_t const h_q  = 4;
    int64_t const h_kv = 2;
    int64_t const s_q  = 1024;
    int64_t const s_kv = 2048;
    int64_t const d    = 128;

    // ---- Override shape configuration (batch up, seq lengths down) ----
    int64_t const new_b    = 4;
    int64_t const new_s_q  = 512;
    int64_t const new_s_kv = 1024;

    float const attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

    if (!is_oss_supported_arch()) {
        SKIP("Test requires Hopper (SM90) or Blackwell (SM100) architecture");
        return;
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // ================================================================
    // Phase 1: Build OSS graph with initial shapes and execute
    // ================================================================

    int64_t const q_elems_init   = b * h_q * s_q * d;
    int64_t const kv_elems_init  = b * h_kv * s_kv * d;
    int64_t const o_elems_init   = b * h_q * s_q * d;
    int64_t const aux_elems_init = b * h_q * s_q;

    Surface<half> q_init(q_elems_init, false);
    Surface<half> k_init(kv_elems_init, false);
    Surface<half> v_init(kv_elems_init, false);
    Surface<half> o_oss_init(o_elems_init, false);
    Surface<float> max_oss_init(aux_elems_init, false);
    Surface<float> se_oss_init(aux_elems_init, false);

    // Build OSS graph (kept alive for re-execution with overrides)
    auto oss_graph = create_sdpa_forward_graph(b, h_q, h_kv, s_q, s_kv, d, attn_scale);
    oss_graph->set_override_shape_enabled(true);

    REQUIRE(oss_graph->validate().is_good());
    REQUIRE(oss_graph->build_operation_graph(handle).is_good());
    REQUIRE(oss_graph->create_execution_plans({fe::HeurMode_t::OPENSOURCE}).is_good());
    REQUIRE(oss_graph->check_support(handle).is_good());
    REQUIRE(oss_graph->build_plans(handle).is_good());

    {
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
            {Q_UID, q_init.devPtr},
            {K_UID, k_init.devPtr},
            {V_UID, v_init.devPtr},
            {O_UID, o_oss_init.devPtr},
            {MAX_UID, max_oss_init.devPtr},
            {SUM_EXP_UID, se_oss_init.devPtr},
        };
        int64_t ws = 0;
        REQUIRE(oss_graph->get_workspace_size(ws).is_good());
        Surface<int8_t> workspace(ws, false);
        auto status = oss_graph->execute(handle, vp, workspace.devPtr);
        if (status.is_bad()) {
            std::printf("OSS initial execute error: %s\n", status.get_message().c_str());
        }
        REQUIRE(status.is_good());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- cuDNN reference for initial shapes ----
    Surface<half> o_ref_init(o_elems_init, false);
    Surface<float> max_ref_init(aux_elems_init, false);
    Surface<float> se_ref_init(aux_elems_init, false);

    {
        auto ref_graph = create_sdpa_forward_graph(b, h_q, h_kv, s_q, s_kv, d, attn_scale);
        REQUIRE(ref_graph->validate().is_good());
        REQUIRE(ref_graph->build_operation_graph(handle).is_good());
        REQUIRE(ref_graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
        REQUIRE(ref_graph->check_support(handle).is_good());
        REQUIRE(ref_graph->build_plans(handle).is_good());

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
            {Q_UID, q_init.devPtr},
            {K_UID, k_init.devPtr},
            {V_UID, v_init.devPtr},
            {O_UID, o_ref_init.devPtr},
            {MAX_UID, max_ref_init.devPtr},
            {SUM_EXP_UID, se_ref_init.devPtr},
        };
        int64_t ws = 0;
        REQUIRE(ref_graph->get_workspace_size(ws).is_good());
        Surface<int8_t> workspace(ws, false);
        REQUIRE(ref_graph->execute(handle, vp, workspace.devPtr).is_good());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    compare_half_outputs(
        o_oss_init.devPtr, o_ref_init.devPtr, o_elems_init, 5e-2, 5e-2, "Initial shapes (OSS vs cuDNN)");

    // ================================================================
    // Phase 2: Re-execute the SAME OSS graph with overridden shapes
    // ================================================================

    int64_t const q_elems_ov   = new_b * h_q * new_s_q * d;
    int64_t const kv_elems_ov  = new_b * h_kv * new_s_kv * d;
    int64_t const o_elems_ov   = new_b * h_q * new_s_q * d;
    int64_t const aux_elems_ov = new_b * h_q * new_s_q;

    Surface<half> q_ov(q_elems_ov, false);
    Surface<half> k_ov(kv_elems_ov, false);
    Surface<half> v_ov(kv_elems_ov, false);
    Surface<half> o_oss_ov(o_elems_ov, false);
    Surface<float> max_oss_ov(aux_elems_ov, false);
    Surface<float> se_oss_ov(aux_elems_ov, false);

    {
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
            {Q_UID, q_ov.devPtr},
            {K_UID, k_ov.devPtr},
            {V_UID, v_ov.devPtr},
            {O_UID, o_oss_ov.devPtr},
            {MAX_UID, max_oss_ov.devPtr},
            {SUM_EXP_UID, se_oss_ov.devPtr},
        };

        // Override shapes: (B, H, S, D) dims with BSHD strides
        std::vector<int64_t> override_uids = {Q_UID, K_UID, V_UID, O_UID, MAX_UID, SUM_EXP_UID};

        std::vector<std::vector<int64_t>> override_shapes = {
            {new_b, h_q, new_s_q, d},    // Q
            {new_b, h_kv, new_s_kv, d},  // K
            {new_b, h_kv, new_s_kv, d},  // V
            {new_b, h_q, new_s_q, d},    // O
            {new_b, h_q, new_s_q, 1},    // Max
            {new_b, h_q, new_s_q, 1},    // Sum_exp
        };

        auto q_st  = bshd_strides(h_q, new_s_q, d);
        auto kv_st = bshd_strides(h_kv, new_s_kv, d);
        auto o_st  = bshd_strides(h_q, new_s_q, d);

        std::vector<std::vector<int64_t>> override_strides = {
            q_st,                            // Q
            kv_st,                           // K
            kv_st,                           // V
            o_st,                            // O
            {h_q * new_s_q, new_s_q, 1, 1},  // Max
            {h_q * new_s_q, new_s_q, 1, 1},  // Sum_exp
        };

        int64_t ws = 0;
        REQUIRE(oss_graph->get_workspace_size(ws).is_good());
        Surface<int8_t> workspace(ws, false);

        auto status =
            oss_graph->execute(handle, vp, workspace.devPtr, override_uids, override_shapes, override_strides);
        if (status.is_bad()) {
            std::printf("OSS dynamic-shape execute error: %s\n", status.get_message().c_str());
        }
        REQUIRE(status.is_good());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- cuDNN reference for override shapes (separate graph) ----
    Surface<half> o_ref_ov(o_elems_ov, false);
    Surface<float> max_ref_ov(aux_elems_ov, false);
    Surface<float> se_ref_ov(aux_elems_ov, false);

    {
        auto ref_graph = create_sdpa_forward_graph(new_b, h_q, h_kv, new_s_q, new_s_kv, d, attn_scale);
        REQUIRE(ref_graph->validate().is_good());
        REQUIRE(ref_graph->build_operation_graph(handle).is_good());
        REQUIRE(ref_graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
        REQUIRE(ref_graph->check_support(handle).is_good());
        REQUIRE(ref_graph->build_plans(handle).is_good());

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
            {Q_UID, q_ov.devPtr},
            {K_UID, k_ov.devPtr},
            {V_UID, v_ov.devPtr},
            {O_UID, o_ref_ov.devPtr},
            {MAX_UID, max_ref_ov.devPtr},
            {SUM_EXP_UID, se_ref_ov.devPtr},
        };
        int64_t ws = 0;
        REQUIRE(ref_graph->get_workspace_size(ws).is_good());
        Surface<int8_t> workspace(ws, false);
        REQUIRE(ref_graph->execute(handle, vp, workspace.devPtr).is_good());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    compare_half_outputs(
        o_oss_ov.devPtr, o_ref_ov.devPtr, o_elems_ov, 5e-2, 5e-2, "Override shapes (OSS dynamic vs cuDNN)");
}

// ---------------------------------------------------------------------------
// TEST_CASE: SM90 Direct engine API -- instantiate Sm90SdpaPrefillEngine
// directly (bypassing graph API), build once, execute with two different
// shape configs.
// ---------------------------------------------------------------------------
TEST_CASE("SM90 Prefill OSS Engine Direct API", "[graph][sdpa][sm90][oss][direct]") {
    int64_t const d = 128;

    // Shape config 1
    int64_t const b1    = 2;
    int64_t const h_q1  = 4;
    int64_t const h_kv1 = 2;
    int64_t const s_q1  = 1024;
    int64_t const s_kv1 = 2048;

    // Shape config 2 (batch up, seq lengths down)
    int64_t const b2    = 4;
    int64_t const h_q2  = 4;
    int64_t const h_kv2 = 2;
    int64_t const s_q2  = 512;
    int64_t const s_kv2 = 1024;

    float const attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

    auto sm_version = get_compute_capability();

    fe::experimental::AttentionShape_t shape = {
        static_cast<uint32_t>(b1),
        static_cast<uint32_t>(h_q1),
        static_cast<uint32_t>(h_kv1),
        static_cast<uint32_t>(h_kv1),
        static_cast<uint32_t>(s_q1),
        static_cast<uint32_t>(s_kv1),
        static_cast<uint32_t>(d),
        static_cast<uint32_t>(d),
    };

    // ---- Build engine once ----
    fe::experimental::Sm90SdpaPrefillEngine engine;

    if (is_hopper_arch()) {
        REQUIRE(engine.check_support(shape, static_cast<int>(sm_version)).is_good());
        REQUIRE(engine.build().is_good());
    } else {
        REQUIRE(engine.check_support(shape, static_cast<int>(sm_version)).is_bad());
        SKIP("Test requires Hopper (SM90) architecture");
        return;
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    // ---- Get CUdevice and create stream ----
    int device_ordinal = 0;
    CUDA_CHECK(cudaGetDevice(&device_ordinal));
    CUdevice cu_device;
    auto cu_err = fe::experimental::detail::cu_device_get(&cu_device, device_ordinal);
    REQUIRE(cu_err == CUDA_SUCCESS);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int64_t const ws_size = fe::experimental::Sm90SdpaPrefillEngine::get_workspace_size();
    Surface<int8_t> workspace(ws_size, false);

    // ================================================================
    // Config 1: b=2, h_q=4, h_kv=2, s_q=1024, s_kv=2048, d=128
    // ================================================================
    {
        int64_t const q_elems   = b1 * h_q1 * s_q1 * d;
        int64_t const kv_elems  = b1 * h_kv1 * s_kv1 * d;
        int64_t const o_elems   = b1 * h_q1 * s_q1 * d;
        int64_t const aux_elems = b1 * h_q1 * s_q1;

        Surface<half> q_tensor(q_elems, false);
        Surface<half> k_tensor(kv_elems, false);
        Surface<half> v_tensor(kv_elems, false);
        Surface<half> o_engine(o_elems, false);
        Surface<float> max_engine(aux_elems, false);
        Surface<float> se_engine(aux_elems, false);

        auto q_st                   = bshd_strides(h_q1, s_q1, d);
        auto kv_st                  = bshd_strides(h_kv1, s_kv1, d);
        auto o_st                   = bshd_strides(h_q1, s_q1, d);
        std::vector<int64_t> max_st = {h_q1 * s_q1, s_q1, 1};
        std::vector<int64_t> se_st  = {h_q1 * s_q1, s_q1, 1};

        auto exec_status = engine.execute(static_cast<int>(b1),
                                          static_cast<int>(h_q1),
                                          static_cast<int>(h_kv1),
                                          static_cast<int>(s_q1),
                                          static_cast<int>(s_kv1),
                                          static_cast<int>(d),
                                          q_tensor.devPtr,
                                          q_st,
                                          k_tensor.devPtr,
                                          kv_st,
                                          v_tensor.devPtr,
                                          kv_st,
                                          o_engine.devPtr,
                                          o_st,
                                          max_engine.devPtr,
                                          max_st,
                                          se_engine.devPtr,
                                          se_st,
                                          workspace.devPtr,
                                          cu_device,
                                          stream,
                                          attn_scale);
        if (exec_status.is_bad()) {
            std::printf("SM90 Direct API config1 execute error: %s\n", exec_status.get_message().c_str());
        }
        REQUIRE(exec_status.is_good());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ---- Benchmark SM90 Direct API ----
        {
            int const warmup_iters = 10;
            int const bench_iters  = 100;

            for (int i = 0; i < warmup_iters; ++i) {
                (void)engine.execute(static_cast<int>(b1),
                                     static_cast<int>(h_q1),
                                     static_cast<int>(h_kv1),
                                     static_cast<int>(s_q1),
                                     static_cast<int>(s_kv1),
                                     static_cast<int>(d),
                                     q_tensor.devPtr,
                                     q_st,
                                     k_tensor.devPtr,
                                     kv_st,
                                     v_tensor.devPtr,
                                     kv_st,
                                     o_engine.devPtr,
                                     o_st,
                                     max_engine.devPtr,
                                     max_st,
                                     se_engine.devPtr,
                                     se_st,
                                     workspace.devPtr,
                                     cu_device,
                                     stream,
                                     attn_scale);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < bench_iters; ++i) {
                (void)engine.execute(static_cast<int>(b1),
                                     static_cast<int>(h_q1),
                                     static_cast<int>(h_kv1),
                                     static_cast<int>(s_q1),
                                     static_cast<int>(s_kv1),
                                     static_cast<int>(d),
                                     q_tensor.devPtr,
                                     q_st,
                                     k_tensor.devPtr,
                                     kv_st,
                                     v_tensor.devPtr,
                                     kv_st,
                                     o_engine.devPtr,
                                     o_st,
                                     max_engine.devPtr,
                                     max_st,
                                     se_engine.devPtr,
                                     se_st,
                                     workspace.devPtr,
                                     cu_device,
                                     stream,
                                     attn_scale);
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            float mean_ms = total_ms / static_cast<float>(bench_iters);

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            std::printf("\n[SM90 Direct API]  Mean kernel time: %.4f ms  (over %d iters, %d warmup)\n",
                        mean_ms,
                        bench_iters,
                        warmup_iters);
        }

        // cuDNN reference for config 1
        Surface<half> o_ref(o_elems, false);
        Surface<float> max_ref(aux_elems, false);
        Surface<float> se_ref(aux_elems, false);
        {
            auto ref_graph = create_sdpa_forward_graph(b1, h_q1, h_kv1, s_q1, s_kv1, d, attn_scale);
            REQUIRE(ref_graph->validate().is_good());
            REQUIRE(ref_graph->build_operation_graph(handle).is_good());
            REQUIRE(ref_graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
            REQUIRE(ref_graph->check_support(handle).is_good());
            REQUIRE(ref_graph->build_plans(handle).is_good());

            std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
                {Q_UID, q_tensor.devPtr},
                {K_UID, k_tensor.devPtr},
                {V_UID, v_tensor.devPtr},
                {O_UID, o_ref.devPtr},
                {MAX_UID, max_ref.devPtr},
                {SUM_EXP_UID, se_ref.devPtr},
            };
            int64_t ws = 0;
            REQUIRE(ref_graph->get_workspace_size(ws).is_good());
            Surface<int8_t> ref_ws(ws, false);
            REQUIRE(ref_graph->execute(handle, vp, ref_ws.devPtr).is_good());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        compare_half_outputs(o_engine.devPtr, o_ref.devPtr, o_elems, 5e-2, 5e-2, "SM90 Direct API config 1");
    }

    // ================================================================
    // Config 2: b=4, h_q=4, h_kv=2, s_q=512, s_kv=1024, d=128
    // Same engine instance, no rebuild.
    // ================================================================
    {
        int64_t const q_elems   = b2 * h_q2 * s_q2 * d;
        int64_t const kv_elems  = b2 * h_kv2 * s_kv2 * d;
        int64_t const o_elems   = b2 * h_q2 * s_q2 * d;
        int64_t const aux_elems = b2 * h_q2 * s_q2;

        Surface<half> q_tensor(q_elems, false);
        Surface<half> k_tensor(kv_elems, false);
        Surface<half> v_tensor(kv_elems, false);
        Surface<half> o_engine(o_elems, false);
        Surface<float> max_engine(aux_elems, false);
        Surface<float> se_engine(aux_elems, false);

        auto q_st                   = bshd_strides(h_q2, s_q2, d);
        auto kv_st                  = bshd_strides(h_kv2, s_kv2, d);
        auto o_st                   = bshd_strides(h_q2, s_q2, d);
        std::vector<int64_t> max_st = {h_q2 * s_q2, s_q2, 1};
        std::vector<int64_t> se_st  = {h_q2 * s_q2, s_q2, 1};

        auto exec_status = engine.execute(static_cast<int>(b2),
                                          static_cast<int>(h_q2),
                                          static_cast<int>(h_kv2),
                                          static_cast<int>(s_q2),
                                          static_cast<int>(s_kv2),
                                          static_cast<int>(d),
                                          q_tensor.devPtr,
                                          q_st,
                                          k_tensor.devPtr,
                                          kv_st,
                                          v_tensor.devPtr,
                                          kv_st,
                                          o_engine.devPtr,
                                          o_st,
                                          max_engine.devPtr,
                                          max_st,
                                          se_engine.devPtr,
                                          se_st,
                                          workspace.devPtr,
                                          cu_device,
                                          stream,
                                          attn_scale);
        if (exec_status.is_bad()) {
            std::printf("SM90 Direct API config2 execute error: %s\n", exec_status.get_message().c_str());
        }
        REQUIRE(exec_status.is_good());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // cuDNN reference for config 2
        Surface<half> o_ref(o_elems, false);
        Surface<float> max_ref(aux_elems, false);
        Surface<float> se_ref(aux_elems, false);
        {
            auto ref_graph = create_sdpa_forward_graph(b2, h_q2, h_kv2, s_q2, s_kv2, d, attn_scale);
            REQUIRE(ref_graph->validate().is_good());
            REQUIRE(ref_graph->build_operation_graph(handle).is_good());
            REQUIRE(ref_graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
            REQUIRE(ref_graph->check_support(handle).is_good());
            REQUIRE(ref_graph->build_plans(handle).is_good());

            std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
                {Q_UID, q_tensor.devPtr},
                {K_UID, k_tensor.devPtr},
                {V_UID, v_tensor.devPtr},
                {O_UID, o_ref.devPtr},
                {MAX_UID, max_ref.devPtr},
                {SUM_EXP_UID, se_ref.devPtr},
            };
            int64_t ws = 0;
            REQUIRE(ref_graph->get_workspace_size(ws).is_good());
            Surface<int8_t> ref_ws(ws, false);
            REQUIRE(ref_graph->execute(handle, vp, ref_ws.devPtr).is_good());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        compare_half_outputs(o_engine.devPtr, o_ref.devPtr, o_elems, 5e-2, 5e-2, "SM90 Direct API config 2");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// TEST_CASE: SM100 Direct engine API -- instantiate Sm100SdpaPrefillEngine
// directly (bypassing graph API), build once, execute with two different
// shape configs.
// ---------------------------------------------------------------------------
TEST_CASE("SM100 Prefill OSS Engine Direct API", "[graph][sdpa][sm100][oss][direct]") {
    int64_t const d = 128;

    // Shape config 1
    int64_t const b1    = 2;
    int64_t const h_q1  = 4;
    int64_t const h_kv1 = 2;
    int64_t const s_q1  = 1024;
    int64_t const s_kv1 = 2048;

    // Shape config 2 (batch up, seq lengths down)
    int64_t const b2    = 4;
    int64_t const h_q2  = 4;
    int64_t const h_kv2 = 2;
    int64_t const s_q2  = 512;
    int64_t const s_kv2 = 1024;

    float const attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

    auto sm_version = get_compute_capability();

    fe::experimental::AttentionShape_t shape = {
        static_cast<uint32_t>(b1),
        static_cast<uint32_t>(h_q1),
        static_cast<uint32_t>(h_kv1),
        static_cast<uint32_t>(h_kv1),
        static_cast<uint32_t>(s_q1),
        static_cast<uint32_t>(s_kv1),
        static_cast<uint32_t>(d),
        static_cast<uint32_t>(d),
    };

    // ---- Build engine once ----
    fe::experimental::Sm100SdpaPrefillEngine engine;

    if (is_blackwell_computing_arch()) {
        REQUIRE(engine.check_support(shape, static_cast<int>(sm_version)).is_good());
        REQUIRE(engine.build().is_good());
    } else {
        REQUIRE(engine.check_support(shape, static_cast<int>(sm_version)).is_bad());
        SKIP("Test requires Blackwell (SM10X) architecture");
        return;
    }

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    // ---- Get CUdevice and create stream ----
    int device_ordinal = 0;
    CUDA_CHECK(cudaGetDevice(&device_ordinal));
    CUdevice cu_device;
    auto cu_err = fe::experimental::detail::cu_device_get(&cu_device, device_ordinal);
    REQUIRE(cu_err == CUDA_SUCCESS);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int64_t const ws_size = fe::experimental::Sm100SdpaPrefillEngine::get_workspace_size();
    Surface<int8_t> workspace(ws_size, false);

    // ================================================================
    // Config 1: b=2, h_q=4, h_kv=2, s_q=1024, s_kv=2048, d=128
    // ================================================================
    {
        int64_t const q_elems   = b1 * h_q1 * s_q1 * d;
        int64_t const kv_elems  = b1 * h_kv1 * s_kv1 * d;
        int64_t const o_elems   = b1 * h_q1 * s_q1 * d;
        int64_t const aux_elems = b1 * h_q1 * s_q1;

        Surface<half> q_tensor(q_elems, false);
        Surface<half> k_tensor(kv_elems, false);
        Surface<half> v_tensor(kv_elems, false);
        Surface<half> o_engine(o_elems, false);
        Surface<float> max_engine(aux_elems, false);
        Surface<float> se_engine(aux_elems, false);

        auto q_st                   = bshd_strides(h_q1, s_q1, d);
        auto kv_st                  = bshd_strides(h_kv1, s_kv1, d);
        auto o_st                   = bshd_strides(h_q1, s_q1, d);
        std::vector<int64_t> max_st = {h_q1 * s_q1, s_q1, 1};
        std::vector<int64_t> se_st  = {h_q1 * s_q1, s_q1, 1};

        auto exec_status = engine.execute(static_cast<int>(b1),
                                          static_cast<int>(h_q1),
                                          static_cast<int>(h_kv1),
                                          static_cast<int>(s_q1),
                                          static_cast<int>(s_kv1),
                                          static_cast<int>(d),
                                          q_tensor.devPtr,
                                          q_st,
                                          k_tensor.devPtr,
                                          kv_st,
                                          v_tensor.devPtr,
                                          kv_st,
                                          o_engine.devPtr,
                                          o_st,
                                          max_engine.devPtr,
                                          max_st,
                                          se_engine.devPtr,
                                          se_st,
                                          workspace.devPtr,
                                          cu_device,
                                          stream,
                                          attn_scale);
        if (exec_status.is_bad()) {
            std::printf("SM100 Direct API config1 execute error: %s\n", exec_status.get_message().c_str());
        }
        REQUIRE(exec_status.is_good());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ---- Benchmark SM100 Direct API ----
        {
            int const warmup_iters = 10;
            int const bench_iters  = 100;

            for (int i = 0; i < warmup_iters; ++i) {
                (void)engine.execute(static_cast<int>(b1),
                                     static_cast<int>(h_q1),
                                     static_cast<int>(h_kv1),
                                     static_cast<int>(s_q1),
                                     static_cast<int>(s_kv1),
                                     static_cast<int>(d),
                                     q_tensor.devPtr,
                                     q_st,
                                     k_tensor.devPtr,
                                     kv_st,
                                     v_tensor.devPtr,
                                     kv_st,
                                     o_engine.devPtr,
                                     o_st,
                                     max_engine.devPtr,
                                     max_st,
                                     se_engine.devPtr,
                                     se_st,
                                     workspace.devPtr,
                                     cu_device,
                                     stream,
                                     attn_scale);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < bench_iters; ++i) {
                (void)engine.execute(static_cast<int>(b1),
                                     static_cast<int>(h_q1),
                                     static_cast<int>(h_kv1),
                                     static_cast<int>(s_q1),
                                     static_cast<int>(s_kv1),
                                     static_cast<int>(d),
                                     q_tensor.devPtr,
                                     q_st,
                                     k_tensor.devPtr,
                                     kv_st,
                                     v_tensor.devPtr,
                                     kv_st,
                                     o_engine.devPtr,
                                     o_st,
                                     max_engine.devPtr,
                                     max_st,
                                     se_engine.devPtr,
                                     se_st,
                                     workspace.devPtr,
                                     cu_device,
                                     stream,
                                     attn_scale);
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            float mean_ms = total_ms / static_cast<float>(bench_iters);

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            std::printf("\n[SM100 Direct API]  Mean kernel time: %.4f ms  (over %d iters, %d warmup)\n",
                        mean_ms,
                        bench_iters,
                        warmup_iters);
        }

        // cuDNN reference for config 1
        Surface<half> o_ref(o_elems, false);
        Surface<float> max_ref(aux_elems, false);
        Surface<float> se_ref(aux_elems, false);
        {
            auto ref_graph = create_sdpa_forward_graph(b1, h_q1, h_kv1, s_q1, s_kv1, d, attn_scale);
            REQUIRE(ref_graph->validate().is_good());
            REQUIRE(ref_graph->build_operation_graph(handle).is_good());
            REQUIRE(ref_graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
            REQUIRE(ref_graph->check_support(handle).is_good());
            REQUIRE(ref_graph->build_plans(handle).is_good());

            std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
                {Q_UID, q_tensor.devPtr},
                {K_UID, k_tensor.devPtr},
                {V_UID, v_tensor.devPtr},
                {O_UID, o_ref.devPtr},
                {MAX_UID, max_ref.devPtr},
                {SUM_EXP_UID, se_ref.devPtr},
            };
            int64_t ws = 0;
            REQUIRE(ref_graph->get_workspace_size(ws).is_good());
            Surface<int8_t> ref_ws(ws, false);
            REQUIRE(ref_graph->execute(handle, vp, ref_ws.devPtr).is_good());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        compare_half_outputs(o_engine.devPtr, o_ref.devPtr, o_elems, 5e-2, 5e-2, "SM100 Direct API config 1");
    }

    // ================================================================
    // Config 2: b=4, h_q=4, h_kv=2, s_q=512, s_kv=1024, d=128
    // Same engine instance, no rebuild.
    // ================================================================
    {
        int64_t const q_elems   = b2 * h_q2 * s_q2 * d;
        int64_t const kv_elems  = b2 * h_kv2 * s_kv2 * d;
        int64_t const o_elems   = b2 * h_q2 * s_q2 * d;
        int64_t const aux_elems = b2 * h_q2 * s_q2;

        Surface<half> q_tensor(q_elems, false);
        Surface<half> k_tensor(kv_elems, false);
        Surface<half> v_tensor(kv_elems, false);
        Surface<half> o_engine(o_elems, false);
        Surface<float> max_engine(aux_elems, false);
        Surface<float> se_engine(aux_elems, false);

        auto q_st                   = bshd_strides(h_q2, s_q2, d);
        auto kv_st                  = bshd_strides(h_kv2, s_kv2, d);
        auto o_st                   = bshd_strides(h_q2, s_q2, d);
        std::vector<int64_t> max_st = {h_q2 * s_q2, s_q2, 1};
        std::vector<int64_t> se_st  = {h_q2 * s_q2, s_q2, 1};

        auto exec_status = engine.execute(static_cast<int>(b2),
                                          static_cast<int>(h_q2),
                                          static_cast<int>(h_kv2),
                                          static_cast<int>(s_q2),
                                          static_cast<int>(s_kv2),
                                          static_cast<int>(d),
                                          q_tensor.devPtr,
                                          q_st,
                                          k_tensor.devPtr,
                                          kv_st,
                                          v_tensor.devPtr,
                                          kv_st,
                                          o_engine.devPtr,
                                          o_st,
                                          max_engine.devPtr,
                                          max_st,
                                          se_engine.devPtr,
                                          se_st,
                                          workspace.devPtr,
                                          cu_device,
                                          stream,
                                          attn_scale);
        if (exec_status.is_bad()) {
            std::printf("SM100 Direct API config2 execute error: %s\n", exec_status.get_message().c_str());
        }
        REQUIRE(exec_status.is_good());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // cuDNN reference for config 2
        Surface<half> o_ref(o_elems, false);
        Surface<float> max_ref(aux_elems, false);
        Surface<float> se_ref(aux_elems, false);
        {
            auto ref_graph = create_sdpa_forward_graph(b2, h_q2, h_kv2, s_q2, s_kv2, d, attn_scale);
            REQUIRE(ref_graph->validate().is_good());
            REQUIRE(ref_graph->build_operation_graph(handle).is_good());
            REQUIRE(ref_graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
            REQUIRE(ref_graph->check_support(handle).is_good());
            REQUIRE(ref_graph->build_plans(handle).is_good());

            std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> vp = {
                {Q_UID, q_tensor.devPtr},
                {K_UID, k_tensor.devPtr},
                {V_UID, v_tensor.devPtr},
                {O_UID, o_ref.devPtr},
                {MAX_UID, max_ref.devPtr},
                {SUM_EXP_UID, se_ref.devPtr},
            };
            int64_t ws = 0;
            REQUIRE(ref_graph->get_workspace_size(ws).is_good());
            Surface<int8_t> ref_ws(ws, false);
            REQUIRE(ref_graph->execute(handle, vp, ref_ws.devPtr).is_good());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        compare_half_outputs(o_engine.devPtr, o_ref.devPtr, o_elems, 5e-2, 5e-2, "SM100 Direct API config 2");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}
