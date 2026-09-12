/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cudnn_frontend {

// The one knob vocabulary every plan in the ranked list speaks, backend and
// frontend (python) engines alike: an autotune result is ``(engine_id, knobs)``
// with knobs keyed by this enum, so the integer values are a persisted contract.
//
//  * ``0 .. FRONTEND_KNOB_TYPE_BASE-1`` mirror ``cudnnBackendKnobType_t`` (mapped
//    by the two converters below; the numbers need not match the backend's).
//  * ``FRONTEND_KNOB_TYPE_BASE ..`` are frontend-only knobs, for tuning axes the
//    backend has no word for. They never reach the backend
//    (``convert_to_backend_knob_type`` refuses them).
//
// Both bands are APPEND-ONLY: never insert, renumber, or reuse a value.
enum class KnobType_t : int64_t {
    NOT_SET = 0,

    SWIZZLE               = 1,
    TILE_SIZE             = 2,
    EDGE                  = 3,
    MULTIPLY              = 4,
    SPLIT_K_BUF           = 5,
    TILEK                 = 6,
    STAGES                = 7,
    REDUCTION_MODE        = 8,
    SPLIT_K_SLC           = 9,
    IDX_MODE              = 10,
    SPECFILT              = 11,
    KERNEL_CFG            = 12,
    WORKSPACE             = 13,
    TILE_CGA_M            = 14,
    TILE_CGA_N            = 15,
    BLOCK_SIZE            = 16,
    OCCUPANCY             = 17,
    ARRAY_SIZE_PER_THREAD = 18,
    SPLIT_COLS            = 19,
    TILE_ROWS             = 20,
    TILE_COLS             = 21,
    LOAD_SIZE             = 22,
    CTA_COUNT             = 23,
    STREAM_K              = 24,
    SPLIT_P_SLC           = 25,
    TILE_M                = 26,
    TILE_N                = 27,
    WARP_SPEC_CFG         = 28,
    SWAP_AB               = 29,
    INPUT_TMA_ENABLE      = 30,
    OUTPUT_TMA_ENABLE     = 31,
    TILE_CGA              = 32,

    // ---- frontend-only band -------------------------------------------------
    // Tile-scheduler policy (an engine-declared enumeration, e.g. natural /
    // LPT / LPT-L2 for the FROST SDPA engines).
    SCHED_POLICY = 1000,
    // Pack the query heads of a GQA/MQA group into one tile (0/1).
    PACK_GQA = 1001,
    // Number of KV chunks each Q tile is split across, recombined afterwards;
    // 1 = off. The backend's counterpart is the on/off STREAM_K mode, a
    // different mechanism, hence a distinct knob.
    SPLIT_KV = 1002,
    // Knobs are performance-only: a plan must compute the same function
    // whichever knob values it runs with, so an autotuner may pick any of
    // them. Anything that changes numerics (e.g. a reduced-precision softmax
    // accumulator) is a graph attribute / numerical note, never a knob.
};

// First value of the frontend-only band; everything below mirrors the backend.
constexpr int64_t FRONTEND_KNOB_TYPE_BASE = 1000;

inline constexpr bool
is_frontend_knob_type(KnobType_t const knob_type) {
    return static_cast<int64_t>(knob_type) >= FRONTEND_KNOB_TYPE_BASE;
}

// The persisted values: a renumbering here silently re-targets every stored
// (engine_id, knobs) record downstream.
static_assert(static_cast<int64_t>(KnobType_t::TILE_CGA) == 32, "backend-mirror knob values are append-only");
static_assert(static_cast<int64_t>(KnobType_t::SCHED_POLICY) == FRONTEND_KNOB_TYPE_BASE,
              "frontend-only knobs start at FRONTEND_KNOB_TYPE_BASE");

class Knob {
   public:
    KnobType_t type  = KnobType_t::NOT_SET;
    int64_t maxValue = 0;
    int64_t minValue = 0;
    int64_t stride   = 0;

    Knob(KnobType_t type, int64_t max, int64_t min, int64_t str)
        : type(type), maxValue(max), minValue(min), stride(str) {}
};

static inline cudnnStatus_t
convert_to_backend_knob_type(KnobType_t const knob_type, cudnnBackendKnobType_t& cudnn_knob_type) {
    // Frontend-only knobs have no backend counterpart by construction; a caller
    // handing one to a backend engine gets a loud NOT_SUPPORTED, never a
    // silently mis-mapped knob.
    if (is_frontend_knob_type(knob_type)) {
        return cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED;
    }
    switch (knob_type) {
        case KnobType_t::SWIZZLE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SWIZZLE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_SIZE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_SIZE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::EDGE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_EDGE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::MULTIPLY:
            cudnn_knob_type = CUDNN_KNOB_TYPE_MULTIPLY;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::SPLIT_K_BUF:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPLIT_K_BUF;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILEK:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILEK;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::STAGES:
            cudnn_knob_type = CUDNN_KNOB_TYPE_STAGES;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::REDUCTION_MODE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_REDUCTION_MODE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::SPLIT_K_SLC:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPLIT_K_SLC;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::IDX_MODE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_IDX_MODE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::SPECFILT:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPECFILT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::KERNEL_CFG:
            cudnn_knob_type = CUDNN_KNOB_TYPE_KERNEL_CFG;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::WORKSPACE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_WORKSPACE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#if (CUDNN_VERSION >= 8600)
        case KnobType_t::TILE_CGA_M:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_CGA_M;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_CGA_N:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_CGA_N;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 8800)
        case KnobType_t::BLOCK_SIZE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_BLOCK_SIZE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 8900)
        case KnobType_t::OCCUPANCY:
            cudnn_knob_type = CUDNN_KNOB_TYPE_OCCUPANCY;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::ARRAY_SIZE_PER_THREAD:
            cudnn_knob_type = CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 8905)
        case KnobType_t::SPLIT_COLS:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPLIT_COLS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_ROWS:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_ROWS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_COLS:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_COLS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::LOAD_SIZE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_LOAD_SIZE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 90700)
        case KnobType_t::CTA_COUNT:
            cudnn_knob_type = CUDNN_KNOB_TYPE_CTA_COUNT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::STREAM_K:
            cudnn_knob_type = CUDNN_KNOB_TYPE_STREAM_K;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::SPLIT_P_SLC:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SPLIT_P_SLC;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_M:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_M;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::TILE_N:
            cudnn_knob_type = CUDNN_KNOB_TYPE_TILE_N;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::WARP_SPEC_CFG:
            cudnn_knob_type = CUDNN_KNOB_TYPE_WARP_SPEC_CFG;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 91800)
        case KnobType_t::SWAP_AB:
            cudnn_knob_type = CUDNN_KNOB_TYPE_SWAP_AB;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 92200)
        case KnobType_t::INPUT_TMA_ENABLE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_INPUT_TMA_ENABLE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case KnobType_t::OUTPUT_TMA_ENABLE:
            cudnn_knob_type = CUDNN_KNOB_TYPE_OUTPUT_TMA_ENABLE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
        case KnobType_t::TILE_CGA:
            // Deprecated in the backend enum but still reported by some engines'
            // knob queries; the numeric value avoids the deprecation warning.
            cudnn_knob_type = (cudnnBackendKnobType_t)26;  // CUDNN_KNOB_TYPE_TILE_CGA
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

inline KnobType_t
convert_from_backend_knob_type(cudnnBackendKnobType_t cudnn_knob_type) {
    switch (cudnn_knob_type) {
        case CUDNN_KNOB_TYPE_SWIZZLE:
            return KnobType_t::SWIZZLE;
        case CUDNN_KNOB_TYPE_TILE_SIZE:
            return KnobType_t::TILE_SIZE;
        case CUDNN_KNOB_TYPE_EDGE:
            return KnobType_t::EDGE;
        case CUDNN_KNOB_TYPE_MULTIPLY:
            return KnobType_t::MULTIPLY;
        case CUDNN_KNOB_TYPE_SPLIT_K_BUF:
            return KnobType_t::SPLIT_K_BUF;
        case CUDNN_KNOB_TYPE_TILEK:
            return KnobType_t::TILEK;
        case CUDNN_KNOB_TYPE_STAGES:
            return KnobType_t::STAGES;
        case CUDNN_KNOB_TYPE_REDUCTION_MODE:
            return KnobType_t::REDUCTION_MODE;
        case CUDNN_KNOB_TYPE_SPLIT_K_SLC:
            return KnobType_t::SPLIT_K_SLC;
        case CUDNN_KNOB_TYPE_IDX_MODE:
            return KnobType_t::IDX_MODE;
        case CUDNN_KNOB_TYPE_SPECFILT:
            return KnobType_t::SPECFILT;
        case CUDNN_KNOB_TYPE_KERNEL_CFG:
            return KnobType_t::KERNEL_CFG;
        case CUDNN_KNOB_TYPE_WORKSPACE:
            return KnobType_t::WORKSPACE;
#if (CUDNN_VERSION >= 8600)
        case CUDNN_KNOB_TYPE_TILE_CGA_M:
            return KnobType_t::TILE_CGA_M;
        case CUDNN_KNOB_TYPE_TILE_CGA_N:
            return KnobType_t::TILE_CGA_N;
#endif
#if (CUDNN_VERSION >= 8800)
        case CUDNN_KNOB_TYPE_BLOCK_SIZE:
            return KnobType_t::BLOCK_SIZE;
#endif
#if (CUDNN_VERSION >= 8900)
        case CUDNN_KNOB_TYPE_OCCUPANCY:
            return KnobType_t::OCCUPANCY;
        case CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD:
            return KnobType_t::ARRAY_SIZE_PER_THREAD;
#endif
#if (CUDNN_VERSION >= 8905)
        case CUDNN_KNOB_TYPE_SPLIT_COLS:
            return KnobType_t::SPLIT_COLS;
        case CUDNN_KNOB_TYPE_TILE_ROWS:
            return KnobType_t::TILE_ROWS;
        case CUDNN_KNOB_TYPE_TILE_COLS:
            return KnobType_t::TILE_COLS;
        case CUDNN_KNOB_TYPE_LOAD_SIZE:
            return KnobType_t::LOAD_SIZE;
#endif
#if (CUDNN_VERSION >= 90700)
        case CUDNN_KNOB_TYPE_CTA_COUNT:
            return KnobType_t::CTA_COUNT;
        case CUDNN_KNOB_TYPE_STREAM_K:
            return KnobType_t::STREAM_K;
        case CUDNN_KNOB_TYPE_SPLIT_P_SLC:
            return KnobType_t::SPLIT_P_SLC;
        case CUDNN_KNOB_TYPE_TILE_M:
            return KnobType_t::TILE_M;
        case CUDNN_KNOB_TYPE_TILE_N:
            return KnobType_t::TILE_N;
        case CUDNN_KNOB_TYPE_WARP_SPEC_CFG:
            return KnobType_t::WARP_SPEC_CFG;
#endif
#if (CUDNN_VERSION >= 91800)
        case CUDNN_KNOB_TYPE_SWAP_AB:
            return KnobType_t::SWAP_AB;
#endif
#if (CUDNN_VERSION >= 92200)
        case CUDNN_KNOB_TYPE_INPUT_TMA_ENABLE:
            return KnobType_t::INPUT_TMA_ENABLE;
        case CUDNN_KNOB_TYPE_OUTPUT_TMA_ENABLE:
            return KnobType_t::OUTPUT_TMA_ENABLE;
#endif
        case 26:  // CUDNN_KNOB_TYPE_TILE_CGA (deprecated)
            return KnobType_t::TILE_CGA;
        default:
            return KnobType_t::NOT_SET;
    }
}

}  // namespace cudnn_frontend