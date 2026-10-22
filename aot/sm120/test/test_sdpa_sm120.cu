// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Correctness test for libcudnn_frontend_aot_sm120.so through the public header-only API (dlopen).
//
//   test_sdpa_sm120 [--lib <path>] [--filter <substr>] [--quick] [--decode-random N] [--compare-lib <path>]
//
//   prefill   f16/bf16 FROST kernels vs an fp32 reference: the selection table over shapes and GQA ratios (1..12,
//             including tile_q % G != 0), and every prebuilt knob set pinned on two shapes each
//   decode    q8_0/q4_0 KV vs a double-precision reference: directed + random cases over the envelope
//   graph     execute() captured into a CUDA graph, replayed, bit-identical to a direct call
//   streams   two streams with their own workspaces, bit-identical to single-stream results
//   reject    unsupported classes, bad arguments, small workspace
//   --compare-lib <other.so>: run the prefill cases through both libraries and require bit-identical outputs (the
//             in-tree runtime shim vs the CuTeDSL runtime build: aot/sm120/build.py --runtime cute-dsl)
//
// Exits 0 with "SKIPPED" when there is no compute capability 12.0 device or the library cannot be loaded.

#include "cudnn_frontend/experimental/sm120_aot_attention.h"

#include "decode_problem.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace aot = cudnn_frontend::experimental::sm120_aot;

static int g_pass = 0, g_fail = 0;
static const char* g_filter = nullptr;

static void
record(bool ok) {
    (ok ? g_pass : g_fail)++;
}

// ---------------------------------------------------------------------------------------------------------------------
// Device helpers (prefill)
// ---------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ float
hash_uniform(uint64_t i, uint32_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull + seed * 0xD1B54A32D192ED03ull;
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 29;
    return float(uint32_t(x >> 40)) * (2.0f / 16777216.0f) - 1.0f;
}

template <class T>
__device__ __forceinline__ float
to_f(T x);
template <>
__device__ __forceinline__ float
to_f(__half x) {
    return __half2float(x);
}
template <>
__device__ __forceinline__ float
to_f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <class T>
__device__ __forceinline__ T
from_f(float x);
template <>
__device__ __forceinline__ __half
from_f(float x) {
    return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16
from_f(float x) {
    return __float2bfloat16_rn(x);
}
template <>
__device__ __forceinline__ float
from_f<float>(float x) {
    return x;
}

template <class T>
__global__ void
fill_kernel(T* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        p[i] = from_f<T>(amp * hash_uniform(i, seed));
}

struct View {
    int64_t sb, ss, sh;  // elements
};

// One block per (query row, q head, batch); scores in shared memory.
template <class T>
__global__ void
reference_kernel(const T* q,
                 const T* k,
                 const T* v,
                 View qv,
                 View kv,
                 View vv,
                 int s_q,
                 int s_kv,
                 int h_q,
                 int h_kv,
                 int d,
                 float scale,
                 int causal,
                 int window_left,
                 const float* sinks,
                 float* o_ref) {
    extern __shared__ float s[];
    __shared__ float red[32];
    const int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z, hk = h / (h_q / h_kv);
    const T* qrow      = q + b * qv.sb + i * qv.ss + h * qv.sh;
    const int64_t ipos = int64_t(i) + s_kv - s_q;
    float m            = -INFINITY;
    for (int j = threadIdx.x; j < s_kv; j += blockDim.x) {
        bool ok = true;
        if (causal) ok = j <= ipos;
        if (window_left >= 0) ok = ok && j >= ipos - window_left;
        float acc = -INFINITY;
        if (ok) {
            const T* krow = k + b * kv.sb + int64_t(j) * kv.ss + hk * kv.sh;
            acc           = 0.f;
            for (int c = 0; c < d; ++c) acc += to_f(qrow[c]) * to_f(krow[c]);
            acc *= scale;
        }
        s[j] = acc;
        m    = fmaxf(m, acc);
    }
    for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = -INFINITY;
    for (int w = 0; w < int(blockDim.x >> 5); ++w) m = fmaxf(m, red[w]);
    const float sink = sinks ? sinks[h] : -INFINITY;
    const float mm   = fmaxf(m, sink);
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < s_kv; j += blockDim.x) {
        float p = s[j] == -INFINITY ? 0.f : expf(s[j] - mm);
        s[j]    = p;
        sum += p;
    }
    for (int off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    sum = 0.f;
    for (int w = 0; w < int(blockDim.x >> 5); ++w) sum += red[w];
    if (sinks) sum += expf(sink - mm);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    float* orow     = o_ref + ((int64_t(b) * s_q + i) * h_q + h) * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < s_kv; ++j)
            if (s[j] != 0.f) acc += s[j] * to_f(v[b * vv.sb + int64_t(j) * vv.ss + hk * vv.sh + c]);
        orow[c] = acc * inv;
    }
}

template <class T>
__global__ void
compare_kernel(const T* o, View ov, const float* ref, int b, int s_q, int h_q, int d, float* max_err, int* nonfinite) {
    const int64_t n = int64_t(b) * s_q * h_q * d;
    float mx        = 0.f;
    int bad         = 0;
    for (int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; idx < n;
         idx += int64_t(gridDim.x) * blockDim.x) {
        int64_t c = idx % d, r = idx / d;
        int64_t h = r % h_q;
        r /= h_q;
        int64_t i = r % s_q, bb = r / s_q;
        float x = to_f(o[bb * ov.sb + i * ov.ss + h * ov.sh + c]);
        if (!isfinite(x)) {
            bad++;
            continue;
        }
        mx = fmaxf(mx, fabsf(x - ref[idx]));
    }
    atomicMax(reinterpret_cast<int*>(max_err), __float_as_int(mx));
    if (bad) atomicAdd(nonfinite, bad);
}

// ---------------------------------------------------------------------------------------------------------------------
// Prefill cases
// ---------------------------------------------------------------------------------------------------------------------

struct PrefillClass {
    int d, dtype, causal, window_left, sinks;  // dtype: CUDNN_FE_AOT_DTYPE_F16 / BF16
    std::string
    name() const {
        char b[96];
        std::snprintf(b,
                      sizeof b,
                      "d%d_%s_%s%s%s",
                      d,
                      dtype == CUDNN_FE_AOT_DTYPE_BF16 ? "bf16" : "f16",
                      causal ? "causal" : "nomask",
                      window_left >= 0 ? "_swa" : "",
                      sinks ? "_sink" : "");
        return b;
    }
};

struct Shape {
    int b, h_kv, g, s_q, s_kv;
    int layout;  // 0 = compact BSHD; 1 = llama.cpp-like: Q BHSD, K/V a cache with spare capacity, O with a padded head
                 // stride
};

struct Buffers {
    void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *ws = nullptr;
    float *o_ref = nullptr, *sinks = nullptr, *err = nullptr;
    int* cnt   = nullptr;
    size_t q_n = 0, k_n = 0, v_n = 0, o_n = 0, ref_n = 0, ws_n = 0;
};

static void
ensure(void** p, size_t* have, size_t need) {
    if (need <= *have) return;
    if (*p) CK(cudaFree(*p));
    CK(cudaMalloc(p, need));
    *have = need;
}

struct Built {
    aot::SdpaArgs args;
    View qv, kv, vv, ov;
    size_t o_bytes;
};

// Inputs for one case; returns the args (pointers into B) and the element views.
static Built
build_case(const PrefillClass& pc, const Shape& sh, Buffers& B, uint32_t seed) {
    const int d = pc.d, h_kv = sh.h_kv, h_q = sh.h_kv * sh.g, b = sh.b, s_q = sh.s_q, s_kv = sh.s_kv;
    Built r{};
    size_t q_n, k_n, o_n;
    if (sh.layout == 0) {
        r.qv = {int64_t(s_q) * h_q * d, int64_t(h_q) * d, d};
        r.kv = {int64_t(s_kv) * h_kv * d, int64_t(h_kv) * d, d};
        r.ov = r.qv;
        q_n  = size_t(b) * s_q * h_q * d;
        k_n  = size_t(b) * s_kv * h_kv * d;
        o_n  = q_n;
    } else {
        const int64_t cap = s_kv + 40;
        r.qv              = {int64_t(h_q) * s_q * d, d, int64_t(s_q) * d};
        r.kv              = {cap * h_kv * d, int64_t(h_kv) * d, d};
        r.ov              = {int64_t(s_q) * h_q * (d + 8), int64_t(h_q) * (d + 8), d + 8};
        q_n               = size_t(b) * s_q * h_q * d;
        k_n               = size_t(b) * cap * h_kv * d;
        o_n               = size_t(b) * s_q * h_q * (d + 8);
    }
    r.vv = r.kv;
    ensure(&B.q, &B.q_n, q_n * 2);
    ensure(&B.k, &B.k_n, k_n * 2);
    ensure(&B.v, &B.v_n, k_n * 2);
    ensure(&B.o, &B.o_n, o_n * 2);
    r.o_bytes     = o_n * 2;
    const bool bf = pc.dtype == CUDNN_FE_AOT_DTYPE_BF16;
    auto fillT    = [&](void* p, size_t n, uint32_t sd, float amp) {
        if (bf)
            fill_kernel<__nv_bfloat16><<<1024, 256>>>(static_cast<__nv_bfloat16*>(p), n, sd, amp);
        else
            fill_kernel<__half><<<1024, 256>>>(static_cast<__half*>(p), n, sd, amp);
    };
    fillT(B.q, q_n, seed + 1, 1.5f);
    fillT(B.k, k_n, seed + 2, 1.5f);
    fillT(B.v, k_n, seed + 3, 1.0f);
    fillT(B.o, o_n, seed + 4, 100.f);  // garbage the kernel must overwrite (and leave the padding alone)
    if (!B.sinks) CK(cudaMalloc(&B.sinks, 4096 * 4));
    fill_kernel<float><<<16, 256>>>(B.sinks, 4096, seed + 5, 3.0f);
    CK(cudaGetLastError());
    aot::SdpaArgs& a = r.args;
    a                = aot::make_args();
    a.b = b, a.h_q = h_q, a.h_kv = h_kv, a.s_q = s_q, a.s_kv = s_kv;
    a.q = B.q, a.k = B.k, a.v = B.v, a.o = B.o;
    const View* views[4] = {&r.qv, &r.kv, &r.vv, &r.ov};
    int64_t* strides[4]  = {a.q_stride, a.k_stride, a.v_stride, a.o_stride};
    for (int t = 0; t < 4; ++t) {
        strides[t][0] = views[t]->sb * 2;
        strides[t][1] = views[t]->ss * 2;
        strides[t][2] = views[t]->sh * 2;
    }
    a.sinks = pc.sinks ? B.sinks : nullptr;
    return r;
}

struct CaseResult {
    aot::Status status;
    std::string msg;
    float max_err;
    int nonfinite;
};

static CaseResult
check_case(const PrefillClass& pc, const Shape& sh, const Built& c, Buffers& B) {
    const aot::SdpaArgs& a = c.args;
    const int d = pc.d, h_q = a.h_q;
    const size_t ref_n = size_t(a.b) * a.s_q * h_q * d;
    ensure(reinterpret_cast<void**>(&B.o_ref), &B.ref_n, ref_n * 4);
    const float scale = 1.f / std::sqrt(float(d));
    const dim3 grid(a.s_q, h_q, a.b);
    const size_t smem = size_t(a.s_kv) * 4;
    if (pc.dtype == CUDNN_FE_AOT_DTYPE_BF16)
        reference_kernel<__nv_bfloat16><<<grid, 128, smem>>>(static_cast<const __nv_bfloat16*>(B.q),
                                                             static_cast<const __nv_bfloat16*>(B.k),
                                                             static_cast<const __nv_bfloat16*>(B.v),
                                                             c.qv,
                                                             c.kv,
                                                             c.vv,
                                                             a.s_q,
                                                             a.s_kv,
                                                             h_q,
                                                             a.h_kv,
                                                             d,
                                                             scale,
                                                             pc.causal,
                                                             pc.window_left,
                                                             a.sinks,
                                                             B.o_ref);
    else
        reference_kernel<__half><<<grid, 128, smem>>>(static_cast<const __half*>(B.q),
                                                      static_cast<const __half*>(B.k),
                                                      static_cast<const __half*>(B.v),
                                                      c.qv,
                                                      c.kv,
                                                      c.vv,
                                                      a.s_q,
                                                      a.s_kv,
                                                      h_q,
                                                      a.h_kv,
                                                      d,
                                                      scale,
                                                      pc.causal,
                                                      pc.window_left,
                                                      a.sinks,
                                                      B.o_ref);
    CK(cudaGetLastError());
    if (!B.err) {
        CK(cudaMalloc(&B.err, 8));
        CK(cudaMalloc(&B.cnt, 8));
    }
    CK(cudaMemset(B.err, 0, 8));
    CK(cudaMemset(B.cnt, 0, 8));
    if (pc.dtype == CUDNN_FE_AOT_DTYPE_BF16)
        compare_kernel<__nv_bfloat16>
            <<<512, 256>>>(static_cast<const __nv_bfloat16*>(B.o), c.ov, B.o_ref, a.b, a.s_q, h_q, d, B.err, B.cnt);
    else
        compare_kernel<__half>
            <<<512, 256>>>(static_cast<const __half*>(B.o), c.ov, B.o_ref, a.b, a.s_q, h_q, d, B.err, B.cnt);
    CK(cudaDeviceSynchronize());
    CaseResult r{};
    CK(cudaMemcpy(&r.max_err, B.err, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&r.nonfinite, B.cnt, 4, cudaMemcpyDeviceToHost));
    if (sh.layout == 1) {  // the padded tail of each O head row must be untouched (filled with |x| <= 100 garbage)
        std::vector<uint16_t> row(c.ov.sh);
        CK(cudaMemcpy(row.data(), B.o, row.size() * 2, cudaMemcpyDeviceToHost));
        const bool bf = pc.dtype == CUDNN_FE_AOT_DTYPE_BF16;
        for (int e = d; e < int(c.ov.sh); ++e) {
            float x;
            if (bf) {
                uint32_t u = uint32_t(row[e]) << 16;
                std::memcpy(&x, &u, 4);
            } else {
                x = __half2float(*reinterpret_cast<const __half*>(&row[e]));
            }
            if (!(std::fabs(x) <= 100.f)) r.nonfinite += 1000000;
        }
    }
    return r;
}

static void
ensure_workspace(Buffers& B, size_t need) {
    ensure(&B.ws, &B.ws_n, std::max<size_t>(need, 256));
}

static bool
run_prefill_case(const char* tag,
                 const PrefillClass& pc,
                 const aot::SdpaPlan& plan,
                 const Shape& sh,
                 Buffers& B,
                 uint32_t seed) {
    Built c           = build_case(pc, sh, B, seed);
    size_t ws         = 0;
    aot::Status st    = plan.workspace_size(c.args, &ws);
    const char* kname = "?";
    aot::SdpaKnobs kn = aot::make_knobs();
    if (!st) st = plan.query(c.args, &kn, &kname);
    std::string kernel = kname;
    CaseResult r{};
    if (!st) {
        ensure_workspace(B, ws);
        st = plan.execute(c.args, B.ws, B.ws_n, 0);
    }
    r.status = st;
    if (st)
        r.msg = plan.last_error();
    else
        r = check_case(pc, sh, c, B);
    const float tol = pc.dtype == CUDNN_FE_AOT_DTYPE_BF16 ? 3e-2f : 6e-3f;
    const bool ok   = !st && r.nonfinite == 0 && r.max_err <= tol;
    record(ok);
    const std::string prefix = "cudnn_fe_aot_sm120_sdpa_fwd_";
    if (kernel.compare(0, prefix.size(), prefix) == 0) kernel = kernel.substr(prefix.size());
    std::printf("%s %-7s %-26s b=%d hq=%d hkv=%d sq=%d skv=%d L%d err=%.2e %s%s%s\n",
                ok ? "PASS" : "FAIL",
                tag,
                pc.name().c_str(),
                sh.b,
                sh.h_kv * sh.g,
                sh.h_kv,
                sh.s_q,
                sh.s_kv,
                sh.layout,
                st ? 0.f : r.max_err,
                kernel.c_str(),
                st ? " status: " : "",
                st ? r.msg.c_str() : "");
    return ok;
}

static aot::SdpaDesc
desc_for(const PrefillClass& pc, int gqa_ratio) {
    aot::SdpaDesc d = aot::make_desc();
    d.head_dim      = pc.d;
    d.q_dtype = d.kv_dtype = d.o_dtype = pc.dtype;
    d.mask                             = pc.causal ? CUDNN_FE_AOT_MASK_CAUSAL : CUDNN_FE_AOT_MASK_NONE;
    d.window_left                      = pc.window_left;
    d.has_sinks                        = pc.sinks;
    d.gqa_ratio                        = gqa_ratio;
    return d;
}

static std::vector<PrefillClass>
prefill_classes() {
    std::vector<PrefillClass> v;
    for (int dt : {CUDNN_FE_AOT_DTYPE_F16, CUDNN_FE_AOT_DTYPE_BF16}) {
        for (int d : {64, 128, 256})
            for (int causal : {0, 1}) v.push_back({d, dt, causal, -1, 0});
        v.push_back({64, dt, 1, -1, 1});
        v.push_back({64, dt, 1, 127, 1});
    }
    return v;
}

static void
run_prefill(bool quick) {
    Buffers B;
    uint32_t seed = 1000;
    for (const PrefillClass& pc : prefill_classes()) {
        if (g_filter && !std::strstr(pc.name().c_str(), g_filter)) continue;
        // 1. Selection table: plan with gqa_ratio 0 (any) over shapes and ratios, incl. G that no PackGQA kernel bakes.
        aot::SdpaPlan any;
        if (aot::Status s = any.build(desc_for(pc, 0))) {
            record(false);
            std::printf("FAIL build   %-26s %s\n", pc.name().c_str(), any.last_error().c_str());
            continue;
        }
        const std::vector<Shape> shapes = {
            {1, 2, 1, 1, 1000, 0},
            {3, 2, 1, 7, 4096, 1},
            {2, 4, 1, 16, 513, 0},
            {1, 2, 1, 300, 300, 0},
            {2, 2, 1, 77, 333, 1},
            {1, 1, 1, 1029, 1029, 1},
            {2, 4, 1, 200, 120, 0},
        };
        const std::vector<int> gs = quick ? std::vector<int>{1, 6} : std::vector<int>{1, 2, 3, 4, 6, 8, 12};
        for (int g : gs)
            for (Shape sh : shapes) {
                sh.g = g;
                run_prefill_case("table", pc, any, sh, B, ++seed);
            }
        // 2. Every prebuilt knob set, pinned.
        std::vector<aot::SdpaKnobs> sets;
        for (int g : {0, 2, 4, 6, 8}) {
            aot::SdpaDesc dg = desc_for(pc, g);
            int32_t n        = 0;
            aot::knob_sets(dg, &n, nullptr, 0);
            std::vector<aot::SdpaKnobs> ks(n);
            aot::knob_sets(dg, &n, ks.data(), n);
            for (const auto& k : ks)
                if ((g == 0) == (k.pack_gqa == 0)) sets.push_back(k);
        }
        for (const aot::SdpaKnobs& k : sets) {
            aot::SdpaPlan pinned;
            const int g = k.pack_gqa ? k.pack_gqa : 4;
            if (aot::Status s = pinned.build(desc_for(pc, k.pack_gqa), &k)) {
                record(false);
                std::printf("FAIL pin     %-26s knobs=(%d,%d,%d,%d,%d) %s\n",
                            pc.name().c_str(),
                            k.pack_gqa,
                            k.sched_policy,
                            k.split_kv,
                            k.tile_m,
                            k.tile_n,
                            pinned.last_error().c_str());
                continue;
            }
            const Shape s1 = k.split_kv > 1 ? Shape{1, 2, g, 1, 1000, 0} : Shape{1, 2, g, 300, 300, 0};
            const Shape s2 = k.split_kv > 1 ? Shape{2, 2, g, 16, 513, 1} : Shape{2, 2, g, 77, 333, 1};
            run_prefill_case("pinned", pc, pinned, s1, B, ++seed);
            if (!quick) run_prefill_case("pinned", pc, pinned, s2, B, ++seed);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode (q8_0 / q4_0 KV), through the unified API
// ---------------------------------------------------------------------------------------------------------------------

static aot::SdpaArgs
decode_args(const Problem& pr) {
    const cudnn_fe_aot_sm120_decode_params_t& p = pr.p;
    aot::SdpaArgs a                             = aot::make_args();
    a.b = p.n_seq, a.h_q = p.n_head, a.h_kv = p.n_head_kv, a.s_q = p.n_q, a.s_kv = p.n_kv;
    a.q = p.q, a.k = p.k, a.v = p.v, a.o = p.o;
    a.q_stride[0] = p.q_nb[3], a.q_stride[1] = p.q_nb[1], a.q_stride[2] = p.q_nb[2];
    a.k_stride[0] = p.k_nb[3], a.k_stride[1] = p.k_nb[1], a.k_stride[2] = p.k_nb[2];
    a.v_stride[0] = p.v_nb[3], a.v_stride[1] = p.v_nb[1], a.v_stride[2] = p.v_nb[2];
    a.o_stride[0] = p.o_nb[3], a.o_stride[1] = p.o_nb[2], a.o_stride[2] = p.o_nb[1];
    a.scale          = p.scale;
    a.mask           = p.mask;
    a.mask_stride[0] = p.mask_nb[3], a.mask_stride[1] = p.mask_nb[1];
    return a;
}

static aot::SdpaDesc
decode_desc(const Case& c) {
    aot::SdpaDesc d = aot::make_desc();
    d.head_dim      = c.D;
    d.q_dtype = d.o_dtype = CUDNN_FE_AOT_DTYPE_F32;
    d.kv_dtype = c.kv_type == CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0 ? CUDNN_FE_AOT_DTYPE_Q8_0 : CUDNN_FE_AOT_DTYPE_Q4_0;
    d.mask     = c.mask ? CUDNN_FE_AOT_MASK_ADDITIVE_F16 : CUDNN_FE_AOT_MASK_NONE;
    return d;
}

static void
run_decode_case(const Case& c, uint32_t seed) {
    Problem pr;
    pr.build(c, seed, true);
    aot::SdpaPlan plan;
    aot::Status st        = plan.build(decode_desc(c));
    const aot::SdpaArgs a = decode_args(pr);
    size_t ws             = 0;
    const char* cfg       = "?";
    void* wsp             = nullptr;
    if (!st) st = plan.workspace_size(a, &ws);
    if (!st) st = plan.query(a, nullptr, &cfg);
    std::string cfgs = cfg;
    if (!st) {
        CK(cudaMalloc(&wsp, std::max<size_t>(ws, 256)));
        st = plan.execute(a, wsp, ws, 0);
    }
    CK(cudaDeviceSynchronize());
    std::vector<float> out((size_t)c.nseq * c.nq * pr.Hq * c.D);
    CK(cudaMemcpy(out.data(), pr.dout, out.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int> heads;
    if (pr.Hq <= 8) {
        for (int h = 0; h < pr.Hq; ++h) heads.push_back(h);
    } else {
        for (int h = 0; h < pr.Hq; h += pr.Hq / 4) heads.push_back(h);
        heads.push_back(pr.Hq - 1);
    }
    double err = 0, ref = 0;
    int nonfinite = 0;
    if (!st) pr.check(out, heads, &err, &ref, &nonfinite);
    const bool ok = !st && nonfinite == 0 && err < 1e-2 * std::max(1.0, ref);
    record(ok);
    static const char* lay[] = {"ggml-test", "kv-cache"};
    static const char* msk[] = {"none", "random", "padded", "causal-inf"};
    std::printf(
        "%s decode  %s D=%d Hkv=%d G=%d kv=%d nq=%d nseq=%d layout=%s mask=%s err=%.2e (|o|max %.2e) [%s]%s%s\n",
        ok ? "PASS" : "FAIL",
        type_name(c.kv_type),
        c.D,
        c.Hkv,
        c.G,
        c.kv,
        c.nq,
        c.nseq,
        lay[c.layout],
        msk[c.mask],
        err,
        ref,
        cfgs.c_str(),
        st ? " status: " : "",
        st ? plan.last_error().c_str() : "");
    if (wsp) CK(cudaFree(wsp));
    pr.release();
}

static void
run_decode(int nrand, uint32_t sd) {
    const int Q8 = CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0, Q4 = CUDNN_FE_AOT_SM120_DECODE_KV_Q4_0;
    std::vector<Case> cases;
    for (int t : {Q8, Q4})
        for (int D : {128, 256}) {
            int i = 0;
            for (int kv : {1, 7, 17, 129, 1000, 4097, 12345}) {
                for (int nq : {1, 3, 8}) {
                    const int G      = D == 128 ? 4 : (i % 2 ? 8 : 2);
                    const int Hkv    = D == 128 ? 2 : (G == 8 ? 2 : 4);
                    const int layout = (i + nq) % 2;
                    const int mask   = kv >= 64 ? (i + nq) % 4 : (i + nq) % 3;
                    cases.push_back({D, Hkv, G, kv, nq, (i % 3 == 0) ? 2 : 1, t, layout, mask});
                }
                ++i;
            }
        }
    std::mt19937 rng(sd);
    for (int n = 0; n < nrand; ++n) {
        Case c;
        c.D            = rng() % 2 ? 128 : 256;
        c.kv_type      = rng() % 2 ? Q8 : Q4;
        const int Gs[] = {1, 2, 4, 5, 6, 8, 12, 16};
        c.G            = Gs[rng() % 8];
        c.Hkv          = 1 + rng() % 4;
        c.nq           = 1 + rng() % 8;
        c.kv           = rng() % 4 == 0 ? 1 + rng() % 64 : 1 + rng() % 20000;
        c.nseq         = rng() % 4 == 0 ? 2 : 1;
        c.layout       = rng() % 2;
        c.mask         = rng() % (c.kv >= 64 ? 4 : 3);
        cases.push_back(c);
    }
    uint32_t seed = sd * 1000;
    for (const Case& c : cases) run_decode_case(c, ++seed);
}

// ---------------------------------------------------------------------------------------------------------------------
// CUDA graph capture, two streams
// ---------------------------------------------------------------------------------------------------------------------

static bool
capture_matches(const aot::SdpaPlan& plan, const aot::SdpaArgs& a, void* out, size_t out_bytes, void* ws, size_t ws_n) {
    std::vector<char> ref(out_bytes), got(out_bytes);
    CK(cudaMemset(
        out, 0xff, out_bytes));  // the same bytes before every run: padding the kernel never writes compares equal
    if (plan.execute(a, ws, ws_n, 0)) return false;
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), out, out_bytes, cudaMemcpyDeviceToHost));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    const aot::Status st = plan.execute(a, ws, ws_n, s);
    CK(cudaStreamEndCapture(s, &g));
    const bool inst = !st && cudaGraphInstantiate(&ge, g, 0) == cudaSuccess;
    bool same       = inst;
    for (int i = 0; i < 3 && same; ++i) {
        CK(cudaMemsetAsync(out, 0xff, out_bytes, s));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(got.data(), out, out_bytes, cudaMemcpyDeviceToHost));
        same = std::memcmp(got.data(), ref.data(), out_bytes) == 0;
    }
    if (inst) cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
    return same;
}

static void
run_graph_and_streams() {
    Buffers B;
    // FROST: an unsplit and a split (decode-shaped) launch.
    const PrefillClass pc{128, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0};
    aot::SdpaPlan plan;
    if (plan.build(desc_for(pc, 4))) {
        record(false);
        std::printf("FAIL graph   build: %s\n", plan.last_error().c_str());
        return;
    }
    for (const Shape& sh : {Shape{1, 8, 4, 512, 512, 1}, Shape{1, 8, 4, 1, 8192, 1}}) {
        Built c   = build_case(pc, sh, B, 7);
        size_t ws = 0;
        plan.workspace_size(c.args, &ws);
        ensure_workspace(B, ws);
        const char* kname = "?";
        plan.query(c.args, nullptr, &kname);
        const bool ok = capture_matches(plan, c.args, B.o, c.o_bytes, B.ws, B.ws_n);
        record(ok);
        std::printf("%s graph   capture + 3 replays bit-identical: %s s_q=%d s_kv=%d\n",
                    ok ? "PASS" : "FAIL",
                    kname,
                    sh.s_q,
                    sh.s_kv);
    }
    // Decode.
    for (const Case& gc :
         {Case{256, 2, 8, 20000, 8, 2, CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0, LAYOUT_KV_CACHE, MASK_KIND_RANDOM},
          Case{128, 8, 4, 3001, 1, 1, CUDNN_FE_AOT_SM120_DECODE_KV_Q4_0, LAYOUT_GGML_TEST, MASK_KIND_NONE}}) {
        Problem pr;
        pr.build(gc, 99, false);
        aot::SdpaPlan dp;
        bool ok = !dp.build(decode_desc(gc));
        if (ok) {
            const aot::SdpaArgs a = decode_args(pr);
            size_t ws             = 0;
            dp.workspace_size(a, &ws);
            void* wsp;
            CK(cudaMalloc(&wsp, std::max<size_t>(ws, 256)));
            ok = capture_matches(dp, a, pr.dout, (size_t)gc.nseq * gc.nq * pr.Hq * gc.D * 4, wsp, ws);
            CK(cudaFree(wsp));
        }
        record(ok);
        std::printf("%s graph   capture + 3 replays bit-identical: decode %s D=%d kv=%d nq=%d\n",
                    ok ? "PASS" : "FAIL",
                    type_name(gc.kv_type),
                    gc.D,
                    gc.kv,
                    gc.nq);
        pr.release();
    }
    // Two streams, two problems, own workspaces, 50 iterations each: outputs match single-stream runs bit for bit.
    {
        Buffers B2;
        const Shape s1{2, 8, 4, 1, 6000, 1}, s2{1, 8, 4, 333, 777, 0};
        Built c1  = build_case(pc, s1, B, 11);
        Built c2  = build_case(pc, s2, B2, 12);
        size_t w1 = 0, w2 = 0;
        plan.workspace_size(c1.args, &w1);
        plan.workspace_size(c2.args, &w2);
        ensure_workspace(B, w1);
        ensure_workspace(B2, w2);
        std::vector<char> r1(c1.o_bytes), r2(c2.o_bytes), g1(c1.o_bytes), g2(c2.o_bytes);
        bool ok = !plan.execute(c1.args, B.ws, B.ws_n, 0) && !plan.execute(c2.args, B2.ws, B2.ws_n, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(r1.data(), B.o, c1.o_bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(r2.data(), B2.o, c2.o_bytes, cudaMemcpyDeviceToHost));
        cudaStream_t sa, sb;
        CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
        for (int it = 0; it < 50 && ok; ++it) {
            ok = !plan.execute(c1.args, B.ws, B.ws_n, sa) && !plan.execute(c2.args, B2.ws, B2.ws_n, sb);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(g1.data(), B.o, c1.o_bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g2.data(), B2.o, c2.o_bytes, cudaMemcpyDeviceToHost));
        ok = ok && !std::memcmp(r1.data(), g1.data(), c1.o_bytes) && !std::memcmp(r2.data(), g2.data(), c2.o_bytes);
        record(ok);
        std::printf("%s streams two streams x 50 iterations bit-identical to single-stream runs\n",
                    ok ? "PASS" : "FAIL");
        cudaStreamDestroy(sa);
        cudaStreamDestroy(sb);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------------------------------------------------

static void
expect(const char* what, aot::Status got, aot::Status want, const std::string& msg) {
    const bool ok = got == want;
    record(ok);
    std::printf("%s reject  %-44s -> %d %s\n", ok ? "PASS" : "FAIL", what, int(got), msg.c_str());
}
// The call runs before last_error() is read (function arguments have no evaluation order).
#define EXPECT(what, call, want, plan)               \
    do {                                             \
        const aot::Status s_ = (call);               \
        expect(what, s_, want, (plan).last_error()); \
    } while (0)

static void
run_rejections() {
    Buffers B;
    const PrefillClass pc{128, CUDNN_FE_AOT_DTYPE_F16, 1, -1, 0};
    {
        aot::SdpaDesc d = desc_for(pc, 0);
        d.head_dim      = 96;
        aot::SdpaPlan p;
        EXPECT("head_dim 96 (not prebuilt)", p.build(d), CUDNN_FE_AOT_STATUS_NOT_SUPPORTED, p);
        d             = desc_for(pc, 0);
        d.window_left = 511;
        EXPECT("causal + window 512 (not prebuilt)", p.build(d), CUDNN_FE_AOT_STATUS_NOT_SUPPORTED, p);
        d         = desc_for(pc, 0);
        d.q_dtype = CUDNN_FE_AOT_DTYPE_F32;
        EXPECT("f32 Q with f16 KV", p.build(d), CUDNN_FE_AOT_STATUS_NOT_SUPPORTED, p);
        aot::SdpaKnobs k = aot::make_knobs();
        k.split_kv = 3, k.tile_m = 64, k.tile_n = 128;
        EXPECT("knob set not prebuilt (split 3)", p.build(desc_for(pc, 0), &k), CUDNN_FE_AOT_STATUS_NOT_SUPPORTED, p);
        k.split_kv = 1, k.pack_gqa = 8, k.tile_m = 128, k.sched_policy = 2;
        EXPECT(
            "PackGQA knobs without gqa_ratio", p.build(desc_for(pc, 0), &k), CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT, p);
        aot::SdpaDesc q = desc_for(pc, 0);
        q.kv_dtype      = CUDNN_FE_AOT_DTYPE_Q8_0;
        EXPECT("q8_0 KV with f16 Q", p.build(q), CUDNN_FE_AOT_STATUS_NOT_SUPPORTED, p);
        q.q_dtype = q.o_dtype = CUDNN_FE_AOT_DTYPE_F32;
        q.head_dim            = 64;
        EXPECT("q8_0 KV head_dim 64", p.build(q), CUDNN_FE_AOT_STATUS_NOT_SUPPORTED, p);
    }
    aot::SdpaPlan plan;
    plan.build(desc_for(pc, 4));
    Built c   = build_case(pc, Shape{1, 2, 4, 1, 4096, 0}, B, 3);
    size_t ws = 0;
    plan.workspace_size(c.args, &ws);
    ensure_workspace(B, ws);
    aot::SdpaArgs a = c.args;
    a.h_q           = 2 * a.h_kv;
    EXPECT(
        "h_q / h_kv != plan gqa_ratio", plan.execute(a, B.ws, B.ws_n, 0), CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT, plan);
    a = c.args;
    a.k_stride[1] += 2;
    EXPECT("K seq stride not a multiple of 16 bytes",
           plan.execute(a, B.ws, B.ws_n, 0),
           CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT,
           plan);
    a   = c.args;
    a.q = static_cast<const char*>(a.q) + 2;
    EXPECT("Q pointer 2-byte aligned", plan.execute(a, B.ws, B.ws_n, 0), CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT, plan);
    a       = c.args;
    a.sinks = B.sinks;
    EXPECT(
        "sinks on a plan without sinks", plan.execute(a, B.ws, B.ws_n, 0), CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT, plan);
    if (ws)
        EXPECT("workspace too small (split launch)",
               plan.execute(c.args, B.ws, ws - 256, 0),
               CUDNN_FE_AOT_STATUS_WORKSPACE_TOO_SMALL,
               plan);
    a     = c.args;
    a.s_q = 0;
    EXPECT("s_q = 0", plan.execute(a, B.ws, B.ws_n, 0), CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT, plan);
    {
        Problem pr;
        pr.build({128, 2, 4, 64, 1, 1, CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0, LAYOUT_GGML_TEST, MASK_KIND_NONE}, 7, false);
        aot::SdpaPlan dp;
        dp.build(
            decode_desc({128, 2, 4, 64, 1, 1, CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0, LAYOUT_GGML_TEST, MASK_KIND_NONE}));
        aot::SdpaArgs da = decode_args(pr);
        da.s_q           = 9;
        size_t dws       = 0;
        EXPECT("q8_0 decode with s_q = 9", dp.workspace_size(da, &dws), CUDNN_FE_AOT_STATUS_NOT_SUPPORTED, dp);
        pr.release();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Two libraries, bit for bit
// ---------------------------------------------------------------------------------------------------------------------

static int
run_compare(aot::Library& other) {
    Buffers B;
    int same = 0, total = 0;
    for (const PrefillClass& pc : prefill_classes()) {
        if (g_filter && !std::strstr(pc.name().c_str(), g_filter)) continue;
        for (int g : {1, 6, 8}) {
            aot::SdpaPlan a, b;
            if (a.build(desc_for(pc, g)) || b.build(desc_for(pc, g), nullptr, -1, &other)) {
                std::printf("FAIL compare build %s: %s / %s\n",
                            pc.name().c_str(),
                            a.last_error().c_str(),
                            b.last_error().c_str());
                record(false);
                continue;
            }
            for (const Shape& sh :
                 {Shape{1, 2, g, 1, 3000, 1}, Shape{2, 2, g, 9, 777, 0}, Shape{1, 2, g, 333, 333, 1}}) {
                Built c   = build_case(pc, sh, B, 77 + g);
                size_t ws = 0;
                a.workspace_size(c.args, &ws);
                ensure_workspace(B, ws);
                std::vector<char> ra(c.o_bytes), rb(c.o_bytes);
                CK(cudaMemset(B.o, 0, c.o_bytes));  // the same bytes before both runs (a padded O keeps its padding)
                bool ok = !a.execute(c.args, B.ws, B.ws_n, 0);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(ra.data(), B.o, c.o_bytes, cudaMemcpyDeviceToHost));
                CK(cudaMemset(B.o, 0, c.o_bytes));
                ok = ok && !b.execute(c.args, B.ws, B.ws_n, 0);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(rb.data(), B.o, c.o_bytes, cudaMemcpyDeviceToHost));
                ok                = ok && !std::memcmp(ra.data(), rb.data(), c.o_bytes);
                const char* kname = "?";
                a.query(c.args, nullptr, &kname);
                record(ok);
                same += ok;
                ++total;
                std::printf("%s compare %-26s G=%d sq=%d skv=%d bit-identical: %s\n",
                            ok ? "PASS" : "FAIL",
                            pc.name().c_str(),
                            g,
                            sh.s_q,
                            sh.s_kv,
                            kname);
            }
        }
    }
    std::printf("COMPARE: %d/%d cases bit-identical between the two libraries\n", same, total);
    return same == total ? 0 : 1;
}

int
main(int argc, char** argv) {
    const char* lib_path = nullptr;
    const char* other    = nullptr;
    bool quick           = false;
    int nrand            = 40;
    const char* only     = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--lib") && i + 1 < argc)
            lib_path = argv[++i];
        else if (!std::strcmp(argv[i], "--compare-lib") && i + 1 < argc)
            other = argv[++i];
        else if (!std::strcmp(argv[i], "--filter") && i + 1 < argc)
            g_filter = argv[++i];
        else if (!std::strcmp(argv[i], "--quick"))
            quick = true;
        else if (!std::strcmp(argv[i], "--decode-random") && i + 1 < argc)
            nrand = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--only") && i + 1 < argc)
            only = argv[++i];  // prefill | decode | graph | reject
        else {
            std::fprintf(stderr,
                         "usage: %s [--lib path] [--compare-lib path] [--filter s] [--quick] [--decode-random N] "
                         "[--only part]\n",
                         argv[0]);
            return 2;
        }
    }
    int dev = 0, n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIPPED: no CUDA device\n");
        return 0;
    }
    for (dev = 0; dev < n; ++dev) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, dev));
        if (p.major == 12 && p.minor == 0) break;
    }
    if (dev == n) {
        std::printf("SKIPPED: no compute capability 12.0 device\n");
        return 0;
    }
    CK(cudaSetDevice(dev));
    aot::Library& lib = aot::Library::instance();
    if (lib.load(lib_path)) {
        std::printf("SKIPPED: %s\n", lib.error().c_str());
        return 0;
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device %d: %s (%d SMs); %s\n", dev, prop.name, prop.multiProcessorCount, lib.build_info());
    CK(cudaFuncSetAttribute(reference_kernel<__half>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CK(cudaFuncSetAttribute(reference_kernel<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    if (other) {
        aot::Library second;
        if (second.load(other)) {
            std::printf("FAIL: %s\n", second.error().c_str());
            return 1;
        }
        std::printf("compare against %s\n", second.build_info());
        return run_compare(second);
    }
    auto want = [&](const char* part) { return !only || !std::strcmp(only, part); };
    if (want("prefill")) run_prefill(quick);
    if (want("decode")) run_decode(quick ? 4 : nrand, 1);
    if (want("graph")) run_graph_and_streams();
    if (want("reject")) run_rejections();
    std::printf("SUMMARY: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
