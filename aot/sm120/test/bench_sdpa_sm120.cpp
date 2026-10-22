// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host and kernel time of libcudnn_frontend_aot_sm120.so through the public API.
//
//   bench_sdpa_sm120 [--lib <path>] [--iters N]
//
// Per case: plan build (one time: loads every kernel module the plan can launch), the host time of the first execute()
// after the build and the steady-state host time per execute() (batches of 100 calls, below the launch-queue depth, so
// the CPU never waits on the GPU), and the kernel time per execute() (CUDA events around back-to-back calls).

#include "cudnn_frontend/experimental/sm120_aot_attention.h"

#include "decode_problem.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace aot = cudnn_frontend::experimental::sm120_aot;
using Clock   = std::chrono::steady_clock;

static double
us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

struct Timing {
    double build_ms, first_us, host_us, kernel_us;
};

static Timing
time_plan(const aot::SdpaDesc& desc, const aot::SdpaArgs& a, int iters, std::string* kernel, aot::Status* status) {
    Timing t{};
    aot::SdpaPlan plan;
    const auto t0 = Clock::now();
    *status       = plan.build(desc);
    t.build_ms    = us_since(t0) / 1e3;
    if (*status) {
        *kernel = plan.last_error();
        return t;
    }
    size_t ws = 0;
    plan.workspace_size(a, &ws);
    void* wsp = nullptr;
    CK(cudaMalloc(&wsp, std::max<size_t>(ws, 256)));
    const char* k = "?";
    plan.query(a, nullptr, &k);
    *kernel = k;
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const auto t1 = Clock::now();
    *status       = plan.execute(a, wsp, ws, s);
    t.first_us    = us_since(t1);
    CK(cudaStreamSynchronize(s));
    std::vector<double> batches;
    for (int rep = 0; rep < 11 && !*status; ++rep) {
        const auto tb = Clock::now();
        for (int i = 0; i < 100; ++i) plan.execute(a, wsp, ws, s);
        batches.push_back(us_since(tb) / 100);
        CK(cudaStreamSynchronize(s));
    }
    std::sort(batches.begin(), batches.end());
    t.host_us = batches.empty() ? 0 : batches[batches.size() / 2];
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < iters; ++i) plan.execute(a, wsp, ws, s);
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    t.kernel_us = ms * 1e3 / iters;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(s);
    CK(cudaFree(wsp));
    return t;
}

static void
row(const char* what, const Timing& t, const std::string& kernel, aot::Status st) {
    const std::string prefix = "cudnn_fe_aot_sm120_";
    std::string k            = kernel.compare(0, prefix.size(), prefix) == 0 ? kernel.substr(prefix.size()) : kernel;
    if (st)
        std::printf("%-46s status %d: %s\n", what, int(st), kernel.c_str());
    else
        std::printf(
            "%-46s %9.1f %9.2f %9.2f %10.1f  %s\n", what, t.build_ms, t.first_us, t.host_us, t.kernel_us, k.c_str());
}

int
main(int argc, char** argv) {
    const char* lib_path = nullptr;
    int iters            = 50;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--lib") && i + 1 < argc)
            lib_path = argv[++i];
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [--lib path] [--iters N]\n", argv[0]);
            return 2;
        }
    }
    aot::Library& lib = aot::Library::instance();
    if (lib.load(lib_path)) {
        std::printf("SKIPPED: %s\n", lib.error().c_str());
        return 0;
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12 || prop.minor != 0) {
        std::printf(
            "SKIPPED: device 0 is %s (%d.%d), not compute capability 12.0\n", prop.name, prop.major, prop.minor);
        return 0;
    }
    std::printf("# %s (%d SMs); %s\n", prop.name, prop.multiProcessorCount, lib.build_info());
    {  // warm the clocks
        void* x;
        CK(cudaMalloc(&x, size_t(1) << 30));
        for (int i = 0; i < 50; ++i) CK(cudaMemset(x, i, size_t(1) << 30));
        CK(cudaDeviceSynchronize());
        CK(cudaFree(x));
    }
    std::printf("%-46s %9s %9s %9s %10s  %s\n", "case", "build ms", "1st us", "host us", "kernel us", "kernel");

    // f16 / bf16 KV (FROST kernels): BSHD, causal bottom-right unless noted.
    struct F {
        const char* name;
        int d, dtype, causal, window_left, sinks, b, h_q, h_kv, s_q, s_kv;
    };
    const F fcases[] = {
        {"f16 d128 32/8 decode s_kv=4096", 128, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0, 1, 32, 8, 1, 4096},
        {"f16 d128 32/8 decode s_kv=32768", 128, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0, 1, 32, 8, 1, 32768},
        {"f16 d128 24/4 (G=6) decode s_kv=8192", 128, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0, 1, 24, 4, 1, 8192},
        {"f16 d256 16/8 decode s_kv=8192", 256, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0, 1, 16, 8, 1, 8192},
        {"f16 d64 64/8 sink decode s_kv=8192", 64, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 1, 1, 64, 8, 1, 8192},
        {"f16 d64 64/8 sink swa128 decode s_kv=8192", 64, CUDNN_FE_AOT_DTYPE_F16, 1, 127, 1, 1, 64, 8, 1, 8192},
        {"f16 d128 32/8 prefill 512", 128, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0, 1, 32, 8, 512, 512},
        {"f16 d128 32/8 prefill 4096", 128, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0, 1, 32, 8, 4096, 4096},
        {"bf16 d128 32/8 prefill 4096", 128, CUDNN_FE_AOT_DTYPE_BF16, 1, -1, 0, 1, 32, 8, 4096, 4096},
        {"f16 d256 16/8 prefill 4096", 256, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0, 1, 16, 8, 4096, 4096},
        {"f16 d128 32/8 no mask 2048", 128, CUDNN_FE_AOT_DTYPE_F16, 0, -1, 0, 1, 32, 8, 2048, 2048},
    };
    for (const F& c : fcases) {
        aot::SdpaDesc d = aot::make_desc();
        d.head_dim      = c.d;
        d.q_dtype = d.kv_dtype = d.o_dtype = c.dtype;
        d.mask                             = c.causal ? CUDNN_FE_AOT_MASK_CAUSAL : CUDNN_FE_AOT_MASK_NONE;
        d.window_left                      = c.window_left;
        d.has_sinks                        = c.sinks;
        d.gqa_ratio                        = c.h_q / c.h_kv;
        const size_t qn = size_t(c.b) * c.s_q * c.h_q * c.d, kn = size_t(c.b) * c.s_kv * c.h_kv * c.d;
        void *q, *k, *v, *o;
        float* sinks;
        CK(cudaMalloc(&q, qn * 2));
        CK(cudaMalloc(&k, kn * 2));
        CK(cudaMalloc(&v, kn * 2));
        CK(cudaMalloc(&o, qn * 2));
        CK(cudaMalloc(&sinks, size_t(c.h_q) * 4));
        CK(cudaMemset(q, 0x31, qn * 2));  // 0x3131: a small positive f16 / bf16 value
        CK(cudaMemset(k, 0x30, kn * 2));
        CK(cudaMemset(v, 0x2f, kn * 2));
        CK(cudaMemset(sinks, 0, size_t(c.h_q) * 4));
        aot::SdpaArgs a = aot::make_args();
        a.b = c.b, a.h_q = c.h_q, a.h_kv = c.h_kv, a.s_q = c.s_q, a.s_kv = c.s_kv;
        a.q = q, a.k = k, a.v = v, a.o = o;
        const int64_t qs[3] = {int64_t(c.s_q) * c.h_q * c.d * 2, int64_t(c.h_q) * c.d * 2, int64_t(c.d) * 2};
        const int64_t ks[3] = {int64_t(c.s_kv) * c.h_kv * c.d * 2, int64_t(c.h_kv) * c.d * 2, int64_t(c.d) * 2};
        std::memcpy(a.q_stride, qs, sizeof qs);
        std::memcpy(a.o_stride, qs, sizeof qs);
        std::memcpy(a.k_stride, ks, sizeof ks);
        std::memcpy(a.v_stride, ks, sizeof ks);
        a.sinks = c.sinks ? sinks : nullptr;
        std::string kernel;
        aot::Status st;
        const Timing t = time_plan(d, a, iters, &kernel, &st);
        row(c.name, t, kernel, st);
        CK(cudaFree(q));
        CK(cudaFree(k));
        CK(cudaFree(v));
        CK(cudaFree(o));
        CK(cudaFree(sinks));
    }

    // q8_0 / q4_0 KV (decode kernels): the ggml test-backend-ops layout with a random f16 mask.
    const int Q8 = CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0, Q4 = CUDNN_FE_AOT_SM120_DECODE_KV_Q4_0;
    struct Dq {
        const char* name;
        Case c;
    };
    const Dq dcases[] = {
        {"q8_0 d128 32/8 decode kv=16384", {128, 8, 4, 16384, 1, 1, Q8, LAYOUT_GGML_TEST, MASK_KIND_RANDOM}},
        {"q8_0 d128 32/8 nq=8 kv=16384", {128, 8, 4, 16384, 8, 1, Q8, LAYOUT_GGML_TEST, MASK_KIND_RANDOM}},
        {"q4_0 d256 16/2 decode kv=65536", {256, 2, 8, 65536, 1, 1, Q4, LAYOUT_GGML_TEST, MASK_KIND_RANDOM}},
        {"q8_0 d256 16/8 decode kv=65536", {256, 8, 2, 65536, 1, 1, Q8, LAYOUT_GGML_TEST, MASK_KIND_RANDOM}},
    };
    for (const Dq& dc : dcases) {
        Problem pr;
        pr.build(dc.c, 1, false);
        const cudnn_fe_aot_sm120_decode_params_t& p = pr.p;
        aot::SdpaDesc d                             = aot::make_desc();
        d.head_dim                                  = dc.c.D;
        d.q_dtype = d.o_dtype = CUDNN_FE_AOT_DTYPE_F32;
        d.kv_dtype            = dc.c.kv_type == Q8 ? CUDNN_FE_AOT_DTYPE_Q8_0 : CUDNN_FE_AOT_DTYPE_Q4_0;
        d.mask                = CUDNN_FE_AOT_MASK_ADDITIVE_F16;
        aot::SdpaArgs a       = aot::make_args();
        a.b = p.n_seq, a.h_q = p.n_head, a.h_kv = p.n_head_kv, a.s_q = p.n_q, a.s_kv = p.n_kv;
        a.q = p.q, a.k = p.k, a.v = p.v, a.o = p.o;
        a.q_stride[0] = p.q_nb[3], a.q_stride[1] = p.q_nb[1], a.q_stride[2] = p.q_nb[2];
        a.k_stride[0] = p.k_nb[3], a.k_stride[1] = p.k_nb[1], a.k_stride[2] = p.k_nb[2];
        a.v_stride[0] = p.v_nb[3], a.v_stride[1] = p.v_nb[1], a.v_stride[2] = p.v_nb[2];
        a.o_stride[0] = p.o_nb[3], a.o_stride[1] = p.o_nb[2], a.o_stride[2] = p.o_nb[1];
        a.scale = p.scale, a.mask = p.mask, a.mask_stride[0] = p.mask_nb[3], a.mask_stride[1] = p.mask_nb[1];
        std::string kernel;
        aot::Status st;
        const Timing t = time_plan(d, a, iters, &kernel, &st);
        row(dc.name, t, kernel, st);
        pr.release();
    }
    return 0;
}
