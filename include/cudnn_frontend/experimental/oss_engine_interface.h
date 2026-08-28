/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nvrtc_shim.h"
#include "attention_utils.h"
#include "../graph_helpers.h"
#include "../../cudnn_frontend_shim.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudnn_frontend {
namespace experimental {

// ============================================================
// IOssNormEngine — virtual interface for OSS norm+activation engines.
// Implemented by Sm100RmsNormSiluEngine (SM100/Blackwell).
// Future implementations: Sm90, Sm110, LayerNorm variants, etc.
// ============================================================

// RmsNormSiluDtype is defined in sm100_rms_norm_silu_knobs.h (self-contained,
// also used by standalone tests without the framework). All arch-specific
// engine implementations share this enum.
//
// Forward-declared here; concrete implementations #include the knobs header.
enum class RmsNormSiluDtype : uint8_t;

// Problem shape for RmsNorm+SiLU (extensible for future fields).
struct NormSiluShape_t {
    int C          = 0;               // hidden dimension (columns)
    int num_tokens = 0;               // number of rows
    RmsNormSiluDtype output_dtype{};  // output data type (bf16, fp8, nvfp4)
};

// Optional parameters for FP8 / NVFP4 output modes.
struct RmsNormSiluExtraParams {
    // FP8 output: quantization scale (float*). If nullptr, uses 1.0 from workspace.
    void* fp8_scale     = nullptr;
    void* fp8_scale_inv = nullptr;  // optional: written by kernel if hasScaleInv
    void* fp8_amax      = nullptr;  // optional: written by kernel if hasAmax

    // NVFP4 (1D1X1X) output: row-wise block-scale factors (float*, written by kernel)
    // Shape: [num_tokens, ceil(C / 16)]
    void* nvfp4_scale_row = nullptr;
};

class IOssNormEngine {
   public:
    virtual ~IOssNormEngine() = default;

    // Phase 1: Validate that (shape, sm_version) is supported.
    virtual error_t
    check_support(NormSiluShape_t shape, int sm_version) = 0;

    // Phase 2: NVRTC compile the kernel for the selected config.
    virtual error_t
    build() = 0;

    // Phase 3: Launch the kernel.
    //   input   — [num_tokens, C] bf16 input tensor
    //   output  — [num_tokens, C] output tensor (bf16, fp8, or nvfp4)
    //   weight  — [C] bf16 gamma weights
    //   bias    — [C] bf16 beta bias (can be nullptr)
    //   rows    — num_tokens
    //   cols    — C (hidden dimension)
    //   epsilon — RMSNorm epsilon (for L2Norm equiv: eps_l2 / C)
    //   extra   — optional FP8/NVFP4 params (can be empty)
    virtual error_t
    execute(void* input,
            void* output,
            void* weight,
            void* bias,
            int rows,
            int cols,
            float epsilon,
            void* workspace,
            int device,
            cudaStream_t stream,
            RmsNormSiluExtraParams const& extra = {}) = 0;

    // Workspace size in bytes. The kernel needs an `rs` (inverse RMS) buffer,
    // plus a float for FP8 default scale if no external scale is provided.
    virtual int64_t
    get_workspace_size() const = 0;
};

}  // namespace experimental
}  // namespace cudnn_frontend
