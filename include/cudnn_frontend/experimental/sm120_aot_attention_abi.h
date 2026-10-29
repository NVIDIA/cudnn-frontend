/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * The flat C ABI of libcudnn_frontend_aot_sm120.so: ahead-of-time compiled attention forward kernels for sm_120a
 * (RTX 50xx, RTX PRO Blackwell). Nothing is compiled at run time. A plan fixes a shape class (head dim, dtypes, mask,
 * sink, GQA ratio); execute() takes the run-time extents, strides and pointers and only launches.
 *
 * Every struct starts with `struct_size` (set it to sizeof the struct); structs only ever grow at the end, and the
 * library reads only the fields the caller's struct_size covers. See docs/fe-oss-apis/aot_sm120_attention.md.
 * C++ callers use sm120_aot_attention.h, which dlopens the library.
 */

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDNN_FE_AOT_SM120_ABI_VERSION 1

typedef enum {
    CUDNN_FE_AOT_STATUS_SUCCESS             = 0,
    CUDNN_FE_AOT_STATUS_INVALID_ARGUMENT    = 1, /* null pointer, misalignment, bad stride or shape */
    CUDNN_FE_AOT_STATUS_NOT_SUPPORTED       = 2, /* no prebuilt kernel serves this shape class / knob set / extent */
    CUDNN_FE_AOT_STATUS_WORKSPACE_TOO_SMALL = 3,
    CUDNN_FE_AOT_STATUS_ARCH_MISMATCH       = 4, /* the device is not compute capability 12.0 */
    CUDNN_FE_AOT_STATUS_CUDA_ERROR          = 5, /* a module load or launch returned a CUDA error */
    CUDNN_FE_AOT_STATUS_LIBRARY_NOT_LOADED  = 6, /* C++ loader only: dlopen / dlsym failed */
    CUDNN_FE_AOT_STATUS_ABI_MISMATCH = 7, /* C++ loader only: the library's ABI version is older than this header's */
} cudnn_fe_aot_status_t;

typedef enum {
    CUDNN_FE_AOT_DTYPE_F16  = 0,
    CUDNN_FE_AOT_DTYPE_BF16 = 1,
    CUDNN_FE_AOT_DTYPE_F32  = 2,
    CUDNN_FE_AOT_DTYPE_Q8_0 = 3, /* ggml block_q8_0: 32 values per 34-byte block (f16 scale + 32 x int8) */
    CUDNN_FE_AOT_DTYPE_Q4_0 = 4, /* ggml block_q4_0: 32 values per 18-byte block (f16 scale + 16 bytes of nibbles) */
} cudnn_fe_aot_dtype_t;

typedef enum {
    CUDNN_FE_AOT_MASK_NONE         = 0,
    CUDNN_FE_AOT_MASK_CAUSAL       = 1, /* bottom-right aligned: query i sees keys j <= i + s_kv - s_q */
    CUDNN_FE_AOT_MASK_ADDITIVE_F16 = 2, /* an f16 [b, s_q, s_kv] tensor added to the scores (quantized KV only) */
} cudnn_fe_aot_mask_t;

/* The shape class a plan is built for. */
typedef struct {
    uint32_t struct_size;
    int32_t head_dim;    /* d_qk == d_v */
    int32_t q_dtype;     /* F16 / BF16 (== kv_dtype), or F32 with a quantized KV */
    int32_t kv_dtype;    /* F16 / BF16 -> FROST kernels; Q8_0 / Q4_0 -> quantized-KV decode kernels */
    int32_t o_dtype;     /* == q_dtype */
    int32_t mask;        /* cudnn_fe_aot_mask_t */
    int32_t window_left; /* -1 = unbounded; W keeps keys j >= i' - W (i' = the bottom-right aligned query position) */
    int32_t has_sinks;   /* per-head attention-sink logits (args.sinks) */
    int32_t gqa_ratio;   /* 0: any h_q / h_kv per call (no PackGQA); G > 0: every call has h_q == G * h_kv */
} cudnn_fe_aot_sdpa_desc_t;

/* A FROST knob set. sched_policy: 0 = NATURAL, 1 = LPT, 2 = LPT_L2. pack_gqa: 0 = off, else the packed ratio G. */
typedef struct {
    uint32_t struct_size;
    int32_t pack_gqa;
    int32_t sched_policy;
    int32_t split_kv;
    int32_t tile_m;
    int32_t tile_n;
} cudnn_fe_aot_sdpa_knobs_t;

/*
 * One call. Logical shapes: Q [b, s_q, h_q, d], K and V [b, s_kv, h_kv, d], O [b, s_q, h_q, d]. Strides are in BYTES,
 * ordered {batch, seq, head}; the head dim is contiguous (for Q8_0 / Q4_0 KV, a row of d / 32 blocks). Any order of
 * the three outer dims works (BSHD, BHSD, a ggml view, a KV cache whose seq stride spans every KV head).
 */
typedef struct {
    uint32_t struct_size;
    int32_t b, h_q, h_kv, s_q, s_kv;
    const void* q;
    const void* k;
    const void* v;
    void* o;
    int64_t q_stride[3];
    int64_t k_stride[3];
    int64_t v_stride[3];
    int64_t o_stride[3];
    float scale;            /* softmax scale; 0 means 1 / sqrt(d) */
    const float* sinks;     /* [h_q] fp32 sink logits when the plan has_sinks, else NULL */
    const void* mask;       /* ADDITIVE_F16 plans: [b, s_q, >= s_kv] f16, contiguous along s_kv */
    int64_t mask_stride[2]; /* bytes {batch, query row}; batch stride 0 broadcasts one mask over the batch */
} cudnn_fe_aot_sdpa_args_t;

typedef struct cudnn_fe_aot_sdpa_plan_s cudnn_fe_aot_sdpa_plan_t;

uint32_t
cudnn_fe_aot_sm120_abi_version(void);
/* "<FE commit> cutlass-dsl <version> <n> prefill kernels ..." */
const char*
cudnn_fe_aot_sm120_build_info(void);

/*
 * Builds a plan on `device` (-1: the current device). Loads every kernel module the plan can launch, so execute()
 * never loads code. `knobs` (may be NULL) pins one FROST knob set instead of the built-in selection table.
 */
cudnn_fe_aot_status_t
cudnn_fe_aot_sm120_sdpa_plan_create(const cudnn_fe_aot_sdpa_desc_t* desc,
                                    const cudnn_fe_aot_sdpa_knobs_t* knobs,
                                    int device,
                                    cudnn_fe_aot_sdpa_plan_t** plan);
void
cudnn_fe_aot_sm120_sdpa_plan_destroy(cudnn_fe_aot_sdpa_plan_t* plan);

/* Device workspace execute() needs for these extents (pointers are not read). 256-byte alignment. */
cudnn_fe_aot_status_t
cudnn_fe_aot_sm120_sdpa_workspace_size(const cudnn_fe_aot_sdpa_plan_t* plan,
                                       const cudnn_fe_aot_sdpa_args_t* args,
                                       size_t* bytes);

/*
 * Launches on `stream`: no allocation, no synchronization, no code loading; capturable in a CUDA graph and safe to call
 * concurrently from several threads / streams (each call with its own workspace).
 */
cudnn_fe_aot_status_t
cudnn_fe_aot_sm120_sdpa_execute(const cudnn_fe_aot_sdpa_plan_t* plan,
                                const cudnn_fe_aot_sdpa_args_t* args,
                                void* workspace,
                                size_t workspace_size,
                                cudaStream_t stream);

/* What execute() would launch for these extents: the knob set (FROST plans) and the main kernel's name. */
cudnn_fe_aot_status_t
cudnn_fe_aot_sm120_sdpa_query(const cudnn_fe_aot_sdpa_plan_t* plan,
                              const cudnn_fe_aot_sdpa_args_t* args,
                              cudnn_fe_aot_sdpa_knobs_t* knobs,
                              const char** kernel_name);

/* The compiled knob sets that serve a shape class (to tune over: build one plan per set). */
int32_t
cudnn_fe_aot_sm120_sdpa_num_knob_sets(const cudnn_fe_aot_sdpa_desc_t* desc);
cudnn_fe_aot_status_t
cudnn_fe_aot_sm120_sdpa_get_knob_set(const cudnn_fe_aot_sdpa_desc_t* desc,
                                     int32_t index,
                                     cudnn_fe_aot_sdpa_knobs_t* knobs);

const char*
cudnn_fe_aot_sm120_status_string(cudnn_fe_aot_status_t status);
/* The reason for this thread's last non-success status ("" if none). */
const char*
cudnn_fe_aot_sm120_last_error(void);

#ifdef __cplusplus
}
#endif
