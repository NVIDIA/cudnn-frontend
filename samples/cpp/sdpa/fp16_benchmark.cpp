/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cuda_runtime_api.h>

#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

/*
Run this example by using command:
bin/samples "Cached sdpa"

This example is supposed to be used when executing full models and/or doing multiple iterations.
*/

// Directly use the forward graph builder from the toy example
std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph(int64_t const b,
                          int64_t const h_q,
                          int64_t const h_k,
                          int64_t const h_v,
                          int64_t const s_q,
                          int64_t const s_kv,
                          int64_t const d_qk,
                          int64_t const d_v,
                          float const attn_scale    = 1.0f,
                          bool const generate_stats = true,
                          bool const causal_mask    = false,
                          bool const alibi_mask     = false,
                          bool const padding_mask   = false,
                          bool has_attn_bias        = false);

// Directly use the backward graph builder from the toy example
std::shared_ptr<fe::graph::Graph>
create_sdpa_backward_graph(int64_t const b,
                           int64_t const h_q,
                           int64_t const h_k,
                           int64_t const h_v,
                           int64_t const s_q,
                           int64_t const s_kv,
                           int64_t const d_qk,
                           int64_t const d_v,
                           float const attn_scale    = 1.0f,
                           bool const generate_stats = true,
                           bool const causal_mask    = false,
                           bool const alibi_mask     = false,
                           bool const padding_mask   = false,
                           bool has_attn_bias        = false);

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define BIAS_UID 6
#define SEQ_LEN_Q_UID 7
#define SEQ_LEN_KV_UID 8

#define DO_UID 101
#define DQ_UID 102
#define DK_UID 103
#define DV_UID 104

TEST_CASE("Benchmark sdpa graph API runtimes", "[graph][sdpa][flash]") {
    SKIP("Very long test turned off by default.");

    int64_t b    = 3;     // batch size
    int64_t h_q  = 4;     // head dim
    int64_t h_k  = 4;     // head dim
    int64_t h_v  = 4;     // head dim
    int64_t s_q  = 1024;  // q tensor is padded to this seq length
    int64_t s_kv = 1024;  // k and v tensor is padded to this seq length
    int64_t d_qk = 128;   // hidden dim
    int64_t d_v  = 128;   // hidden dim

    if (cudnnGetVersion() < 8903) {
        SKIP("Test requires cudnn 8.9.3 or above");
        return;
    }

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    BENCHMARK_ADVANCED("Create")(Catch::Benchmark::Chronometer meter) {
        meter.measure([&] { auto g = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v); });
    };

    BENCHMARK_ADVANCED("Validate")(Catch::Benchmark::Chronometer meter) {
        auto g = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);

        meter.measure([&] { return g->validate(); });
    };

    BENCHMARK_ADVANCED("Build Backend Operation Graph")(Catch::Benchmark::Chronometer meter) {
        auto g      = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);
        auto status = g->validate();

        meter.measure([&] { return g->build_operation_graph(handle); });
    };

    BENCHMARK_ADVANCED("Create Execution Plans")(Catch::Benchmark::Chronometer meter) {
        auto g      = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);
        auto status = g->validate();
        status      = g->build_operation_graph(handle);

        meter.measure([&] { return g->create_execution_plans({fe::HeurMode_t::A}); });
    };

    BENCHMARK_ADVANCED("Check Support")(Catch::Benchmark::Chronometer meter) {
        auto g      = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);
        auto status = g->validate();
        status      = g->build_operation_graph(handle);
        status      = g->create_execution_plans({fe::HeurMode_t::A});

        meter.measure([&] { return g->check_support(); });
    };

    BENCHMARK_ADVANCED("Cached Build plan")(Catch::Benchmark::Chronometer meter) {
        auto g      = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);
        auto status = g->validate();
        status      = g->build_operation_graph(handle);
        status      = g->create_execution_plans({fe::HeurMode_t::A});
        status      = g->check_support();
        status      = g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false);

        meter.measure([&] { return g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false); });
    };

    BENCHMARK_ADVANCED("Workspace query")(Catch::Benchmark::Chronometer meter) {
        auto g      = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);
        auto status = g->validate();
        status      = g->build_operation_graph(handle);
        status      = g->create_execution_plans({fe::HeurMode_t::A});
        status      = g->check_support();
        status      = g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false);

        meter.measure([&] { return g->get_workspace_size(); });
    };
}
