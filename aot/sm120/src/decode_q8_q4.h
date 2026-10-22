// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Split-KV flash-decode over a quantized (q8_0 / q4_0) KV cache for SM120 (RTX 50xx).
// Semantics follow ggml GGML_OP_FLASH_ATTN_EXT for the decode regime (few query tokens, long KV):
//   O[s][t][h][:] = softmax(scale * Q[s][h][t][:] . K[s][h/G][j][:] + mask[s][t][j]) @ V[s][h/G][j][:]
// All strides are in BYTES and use ggml's tensor convention (nb[0] = element/block size, nb[1..3] = dims 1..3),
// so a ggml_tensor's ->data / ->nb can be passed straight through.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Values match enum ggml_type.
typedef enum {
    CUDNN_FE_AOT_SM120_DECODE_KV_Q4_0 = 2,
    CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0 = 8,
} cudnn_fe_aot_sm120_decode_kv_type_t;

typedef enum {
    CUDNN_FE_AOT_SM120_DECODE_SUCCESS = 0,
    CUDNN_FE_AOT_SM120_DECODE_ERROR_NULL_POINTER,
    CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_HEAD_DIM,   // head_dim not in {128, 256}
    CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_KV_TYPE,    // K/V not q8_0/q4_0, or K type != V type
    CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_N_Q,        // n_q not in [1, 8]
    CUDNN_FE_AOT_SM120_DECODE_ERROR_BAD_SHAPE,              // n_kv < 1, n_head % n_head_kv != 0, n_seq < 1, ...
    CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ALIGNMENT,  // K/V pointers or strides not 8-byte aligned, Q/O not
                                                            // 4-byte
    CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_FEATURE,    // max_bias (ALiBi), logit_softcap, sinks
    CUDNN_FE_AOT_SM120_DECODE_ERROR_WORKSPACE_TOO_SMALL,
    CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ARCH,  // device is not compute capability 12.0
    CUDNN_FE_AOT_SM120_DECODE_ERROR_CUDA,              // a CUDA runtime call failed (see cudaGetLastError)
} cudnn_fe_aot_sm120_decode_status_t;

typedef struct {
    int head_dim;   // D: 128 or 256 (K and V head size are equal)
    int n_q;        // query tokens per sequence: 1..8
    int n_head;     // query heads (Hq)
    int n_head_kv;  // KV heads (Hkv); n_head % n_head_kv == 0
    int n_kv;       // KV length: any value >= 1
    int n_seq;      // ggml dim 3 (independent sequences / streams), >= 1; looped on the host

    // Q: f32, ggml ne = [D, n_q, n_head, n_seq]
    const float* q;
    int64_t q_nb[4];
    // K, V: ggml block-quantized rows, ne = [D, n_kv, n_head_kv, n_seq]. nb[1] = KV-position stride,
    // nb[2] = KV-head stride. Head-major (nb[1] = row size) and llama.cpp KV-cache token-major
    // (nb[1] = n_head_kv * row size, nb[2] = row size) views are both supported.
    int kv_type;  // cudnn_fe_aot_sm120_decode_kv_type_t, same for K and V
    const void* k;
    int64_t k_nb[4];
    const void* v;
    int64_t v_nb[4];
    // Optional additive mask: f16, ne = [>= n_kv, >= n_q, 1, 1 or n_seq]; NULL = no mask.
    // Broadcast over heads. Set mask_nb[3] = 0 to broadcast over sequences.
    const void* mask;
    int64_t mask_nb[4];
    // O: f32, ggml ne = [D, n_head, n_q, n_seq] (the FLASH_ATTN_EXT dst layout)
    float* o;
    int64_t o_nb[4];

    float scale;          // usually 1/sqrt(D)
    float max_bias;       // must be 0 (ALiBi unsupported)
    float logit_softcap;  // must be 0 (unsupported)
} cudnn_fe_aot_sm120_decode_params_t;

// Validates the inputs without launching anything.
cudnn_fe_aot_sm120_decode_status_t
cudnn_fe_aot_sm120_decode_check(const cudnn_fe_aot_sm120_decode_params_t* p);

// Bytes of device workspace needed by run() on the current device (depends on the SM count). 0 on invalid input.
size_t
cudnn_fe_aot_sm120_decode_get_workspace_size(const cudnn_fe_aot_sm120_decode_params_t* p);

// Enqueues the attention on `stream` (main kernel + combine kernel per sequence). Asynchronous; no host sync,
// no allocation. The tuned kernel configuration is chosen internally from (head_dim, kv_type, n_q * G, n_head_kv).
cudnn_fe_aot_sm120_decode_status_t
cudnn_fe_aot_sm120_decode_run(const cudnn_fe_aot_sm120_decode_params_t* p,
                              void* workspace,
                              size_t workspace_size,
                              cudaStream_t stream);

// Writes a one-line description of the configuration run() would pick (kernel name, split count) into buf.
cudnn_fe_aot_sm120_decode_status_t
cudnn_fe_aot_sm120_decode_describe(const cudnn_fe_aot_sm120_decode_params_t* p, char* buf, size_t buf_size);

const char*
cudnn_fe_aot_sm120_decode_status_string(cudnn_fe_aot_sm120_decode_status_t s);

#ifdef __cplusplus
}
#endif
