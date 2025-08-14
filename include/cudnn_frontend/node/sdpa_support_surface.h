#pragma once

#include <cstdlib>

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"
#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

inline error_t
SDPA_attributes::validate_sdpa_support_surface(const detail::Context& context,
                                               int64_t s_kv,
                                               bool is_paged_k,
                                               bool is_paged_v) const {
    // Extract dimensions from tensors
    int64_t s_q = inputs.at(SDPA_attributes::input_names::Q)->get_dim()[2];
    // s_kv is passed in from the caller
    int64_t h_q  = inputs.at(SDPA_attributes::input_names::Q)->get_dim()[1];
    int64_t h_k  = inputs.at(SDPA_attributes::input_names::K)->get_dim()[1];
    int64_t h_v  = inputs.at(SDPA_attributes::input_names::V)->get_dim()[1];
    int64_t d_qk = inputs.at(SDPA_attributes::input_names::Q)->get_dim()[3];
    int64_t d_v  = inputs.at(SDPA_attributes::input_names::V)->get_dim()[3];

    bool const is_ragged = inputs.at(SDPA_attributes::input_names::Q)->get_ragged_offset() ||
                           inputs.at(SDPA_attributes::input_names::K)->get_ragged_offset() ||
                           inputs.at(SDPA_attributes::input_names::V)->get_ragged_offset() ||
                           outputs.at(SDPA_attributes::output_names::O)->get_ragged_offset();

    auto const& output_tensor    = outputs.at(SDPA_attributes::output_names::O);
    auto const& output_data_type = output_tensor->get_data_type();

    auto const& bias_mask = inputs.find(SDPA_attributes::input_names::Bias);
    bool const is_bias    = (bias_mask != inputs.end() && bias_mask->second != nullptr);

    auto const& dropout_mask     = inputs.find(SDPA_attributes::input_names::Dropout_mask);
    bool const is_dropout_custom = (dropout_mask != inputs.end()) && (dropout_mask->second != nullptr);
    bool const is_dropout        = dropout_probability.has_value() || is_dropout_custom;

    bool const is_paged = is_paged_k || is_paged_v;

    auto const& rng_tensor = outputs.find(SDPA_attributes::output_names::RNG_DUMP);
    bool const is_rng      = (rng_tensor != outputs.end() && rng_tensor->second != nullptr);

    bool const max_seq_kv_explicit = max_seq_len_kv.has_value();

    auto const& attn_scale    = inputs.find(SDPA_attributes::input_names::Attn_scale);
    bool const has_attn_scale = (attn_scale != inputs.end()) && (attn_scale->second != nullptr);

    auto const& seq_len_q     = inputs.find(SDPA_attributes::input_names::SEQ_LEN_Q);
    bool const has_seq_len_q  = (seq_len_q != inputs.end()) && (seq_len_q->second != nullptr);
    auto const& seq_len_kv    = inputs.find(SDPA_attributes::input_names::SEQ_LEN_KV);
    bool const has_seq_len_kv = (seq_len_kv != inputs.end()) && (seq_len_kv->second != nullptr);

    // validation TODO:
    //    - validate stats has valid dims

    // Get device properties
    cudaDeviceProp prop;
    int device;
    _CUDNN_CHECK_CUDA_ERROR(detail::cuda_get_device(&device));
    _CUDNN_CHECK_CUDA_ERROR(detail::cuda_get_device_properties(&prop, device));

    // Common FP16 and FP8 validation
    // validate basic dimension requirements
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        (h_q % h_k != 0) || (h_q % h_v != 0),
        error_code_t::GRAPH_NOT_SUPPORTED,
        "For group-query attention, number of heads for key and query must be a factor of number of heads for query");

    // validate options for attn_scale
    RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attn_scale_value.has_value(),
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "attn_scale with tensor and value cannot be set at the same time.");

    // validate options for bias mask
    RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Bias mask data type cannot be boolean");
    RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && detail::get_backend_version() < 8906,
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Bias mask is not supported below cudnn version 8.9.6");

    RETURN_CUDNN_FRONTEND_ERROR_IF((detail::get_backend_version() >= 8906 && detail::get_backend_version() < 90000) &&
                                       (context.get_sm_version() > 0 && context.get_sm_version() < 90),
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Post scale Bias mask is not supported below Hopper for cudnn version" +
                                       std::to_string(detail::get_backend_version()));

    // validate options for padding mask
    RETURN_CUDNN_FRONTEND_ERROR_IF(padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "Padding mask requires seq_len_q and seq_len_kv to be set.");
    RETURN_CUDNN_FRONTEND_ERROR_IF((!padding_mask && !attention_score_modifier) && (has_seq_len_q || has_seq_len_kv),
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

    RETURN_CUDNN_FRONTEND_ERROR_IF(is_ragged && (padding_mask == false),
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Ragged offsets are only supported with padding mask.");

    // validate options for dropout mask
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        dropout_probability.has_value() && is_dropout_custom,
        error_code_t::ATTRIBUTE_NOT_SET,
        "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

    RETURN_CUDNN_FRONTEND_ERROR_IF(dropout_probability.has_value() && dropout_probability.value() == 1.0,
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

    // validate options for causal mask and bottom right causal mask
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        (padding_mask || alibi_mask || has_causal_mask_bottom_right()) && (detail::get_backend_version() < 8906),
        error_code_t::GRAPH_NOT_SUPPORTED,
        "Only causal mask is supported in cudnn versions below 8.9.6");

    RETURN_CUDNN_FRONTEND_ERROR_IF(
        has_causal_mask_bottom_right() && (!padding_mask) && s_q > s_kv,
        error_code_t::GRAPH_NOT_SUPPORTED,
        "Bottom right causal mask does not support max_s_q > max_s_kv. Please virtually slice the Q tensor and pass it "
        "as max_s_q == max_s_kv");

    RETURN_CUDNN_FRONTEND_ERROR_IF(
        has_causal_mask_bottom_right() && (is_bias || alibi_mask || is_dropout),
        error_code_t::GRAPH_NOT_SUPPORTED,
        "Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False.");

    RETURN_CUDNN_FRONTEND_ERROR_IF(has_causal_mask_bottom_right() && (detail::get_backend_version() < 90600) &&
                                       ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv "
                                   "multiple of 64, for cudnn version below 9.6.0");

    // validate that datatype is set for the graph
    RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "Intermediate tensor data type needs to be set as internal tensors require it.");

    if (mma_core_mode == DataType_t::FP8_E4M3 || mma_core_mode == DataType_t::FP8_E5M2) {
        // FP8 specific validation

        // version specific validation
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90100,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 forward operation is only supported starting cudnn 9.1.0. Please "
                                       "consider upgrading your current version.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() == 91000,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 forward operation is not supported on cudnn 9.10.0. Please "
                                       "consider upgrading your current version.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            prop.major < 9,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Hopper architecture and newer. Please "
            "consider using a newer architecture.");

        // validate basic dimension requirements
        if (prop.major >= 10) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                (d_qk > 128) || (d_qk % 16 != 0) || (d_v > 128) || (d_v % 16 != 0),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "hidden_dim shoud be less than or equal to 128 and hidden_dim should be multiple of 16");
        } else {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                (d_qk > 256) || (d_qk % 16 != 0) || (d_v > 256) || (d_v % 16 != 0),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "hidden_dim shoud be less than or equal to 256 and hidden_dim should be multiple of 16");
        }

        // Validate options for causal_mask_bottom_right
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_causal_mask_bottom_right() && detail::get_backend_version() < 90700,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.7.0, bottom right causal masking is not supported.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            has_causal_mask_bottom_right() && prop.major < 10,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Blackwell architecture and newer. Please "
            "consider using a newer architecture.");

        // if output data type is half or bfloat16, and version is below 9.13 or is not blackwell, return NOT_SUPPORTED
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (output_data_type == DataType_t::HALF || output_data_type == DataType_t::BFLOAT16) &&
                (detail::get_backend_version() < 91300 || prop.major < 10),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on cuDNN version 9.13.0 and newer. Please "
            "consider upgrading your current version.");
    } else if (mma_core_mode == DataType_t::HALF) {
        // FP16 specific validation

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (attention_score_modifier != nullptr) &&
                (alibi_mask || has_causal_like_masking() || padding_mask || left_bound.has_value()),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Attention score mod enabled and hence other subgraphs are disabled.");

        // validate basic dimension requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (d_qk % 8 != 0) || (d_v % 8 != 0), error_code_t::GRAPH_NOT_SUPPORTED, "hidden_dim should be multiple of 8");

        // validate alibi requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF(alibi_mask && !(right_bound.has_value() && right_bound.value() == 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "When alibi mask is used, diagonal_band_right_bound needs to be set to 0.");

        // validate options for bottom right causal mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_causal_mask_bottom_right() && (detail::get_backend_version() < 90300),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Causal bottom right masking requires cudnn 9.3.0 and above");

        // Combination of mask and bias
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (is_bias && (has_causal_like_masking() || padding_mask) && (detail::get_backend_version() < 8906)),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Bias + padding or causal mask is only supported in 8.9.6 and above");

        // validate options for sliding window length
        RETURN_CUDNN_FRONTEND_ERROR_IF((left_bound.has_value() && detail::get_backend_version() < 90200),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sliding window is only supported 9.2.0 and above");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            left_bound.has_value() && left_bound.value() <= 0 && detail::get_backend_version() < 91000,
            error_code_t::INVALID_VALUE,
            "Left bound (Sliding window length) should be greater than zero when set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(left_bound.has_value() && (!padding_mask) && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Sliding window attention is only supported with max_s_q <= max_s_kv.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            left_bound.has_value() && (s_q * left_bound.value() == s_kv * left_bound.value()) &&
                (detail::get_backend_version() <= 90900) && (prop.major == 9) && has_causal_mask_bottom_right(),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "On Hopper architecture, this specific combination of s_q, s_kv, and left_bound + right_bound + bottom "
            "right diagonal alignment is not supported for backend version 9.9 or below");

        if ((detail::get_backend_version() < 91002)) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                left_bound.has_value() && (!has_causal_like_masking() || is_dropout || is_bias),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "Left and right bounds are only supported with is_dropout=False, is_bias=False. And the diagonal "
                "alignment must be set.");
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(right_bound.has_value() && right_bound.value() < 0,
                                       error_code_t::INVALID_VALUE,
                                       "Right bound needs to be larger than or equal to zero");

        // Validate options for s_q == 1
        const bool is_decode_only = (s_q == 1);
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_decode_only && (prop.major == 10) && (d_qk > 128 || d_v > 128) &&
                                           (detail::get_backend_version() <= 90900),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "decode only mode, i.e. s_q == 1 not supported for blackwell architecture with "
                                       "d_qk or d_v > 128 for backend version 9.9 or below");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            is_decode_only && (detail::get_backend_version() <= 90900) && (right_bound.has_value()),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "decode only mode, i.e. s_q == 1, not supported with masking (right_bound is set) for backend version 9.9 "
            "or below");

        // validate options for paged attention
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            is_paged && (d_qk > 128 || d_v > 128) && detail::get_backend_version() <= 90900,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Paged attention only supported with d_qk and d_v <= 128 for backend version 9.9 or below");

        RETURN_CUDNN_FRONTEND_ERROR_IF(is_paged && is_ragged && detail::get_backend_version() < 90700,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Paged caches are not supported in combination with ragged offsets.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(is_paged && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Paged caches can only be used in combination with padding mask and variable "
                                       "sequence lengths for both Q and KV.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !is_paged && max_seq_kv_explicit,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "When not using paged attention, there is no need to explicitly set max kv sequence length.");

        if (max_seq_kv_explicit) {
            auto max_seq_kv = max_seq_len_kv.value();

            RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_dim()[3] != max_seq_kv),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Value set through set_paged_attention_max_seq_len_kv is incompatible with "
                                           "the sequence length of the bias");

            RETURN_CUDNN_FRONTEND_ERROR_IF(is_rng && rng_tensor->second->get_dim()[3] != max_seq_kv,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Value set through set_paged_attention_max_seq_len_kv is incompatible with "
                                           "the sequence length of the RNG_DUMP");
        }

        // Additional validation for paged attention with packed page tables
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            ((is_paged_k && inputs.at(SDPA_attributes::input_names::Page_table_K)->get_ragged_offset()) ||
             (is_paged_v && inputs.at(SDPA_attributes::input_names::Page_table_V)->get_ragged_offset())) &&
                detail::get_backend_version() < 91002,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Paged attention with packed page tables only supported with cudnn version 9.10.2 and above");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8903,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "SDPA OP requires cudnn version 8.9.3 and above");

        // If user has set sm_version allow SM specific checks
        if (context.get_sm_version() > 0) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(80 > context.get_sm_version(),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "cudnn SDPA operation requires Ampere and above");
        }

        // (cudnn_runtime_version < 8907 && num_attn_heads == num_gqa_groups FIXME

        // version specific validation
        if (prop.major == 8) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                detail::get_backend_version() <= 90900 && ((d_qk > 128) || (d_v > 128)),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "head_dim should be less than or equal to 128 for backend version 9.9 or below on ampere architecture");
        }
        if (prop.major == 9) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                detail::get_backend_version() <= 90900 && ((d_qk > 256) || (d_v > 256)),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "head_dim should be less than or equal to 256 for backend version 9.9 or below on hopper architecture");
        }
        if (prop.major == 10) {
            RETURN_CUDNN_FRONTEND_ERROR_IF((detail::get_backend_version() < 90900) && ((d_qk > 128) || (d_v > 128)),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "head_dim should be less than or equal to 128 for backend version 9.8 or "
                                           "below on blackwell architecture");
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() < 8906 && ((s_kv % 64 != 0) || (d_qk % 64 != 0)),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "For cuDNN version below 8.9.6, s_kv not a multiple of 64 or d not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 8907 && (s_kv % 64 != 0) && (!(padding_mask)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 8.9.7, s_kv not a multiple of 64 is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() < 90000 && ((s_q % 64 != 0) || (s_kv % 64 != 0)) &&
                (padding_mask || is_dropout),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "For cuDNN version below 9.0.0, s_q/s_kv not a multiple of 64 with padding/dropout mask is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90200 && left_bound.has_value(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.2.0, sliding window attention is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500 && is_paged,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.5.0, paged caches are not supported");

        if (is_ragged) {
            RETURN_CUDNN_FRONTEND_ERROR_IF((context.get_sm_version() > 0 && context.get_sm_version() < 90),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "THD (ragged offset) is only supported in Hopper and above");
        }
        // TODO add version check once fixed
        RETURN_CUDNN_FRONTEND_ERROR_IF(prop.major == 10 && is_rng,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "dropout RNG dump is not supported for Blackwell architecture");
    } else {
        RETURN_CUDNN_FRONTEND_ERROR_IF(true, error_code_t::GRAPH_NOT_SUPPORTED, "Unsupported mma core mode");
    }

    // Check whether the selected implementation supports the requested features.
    CHECK_CUDNN_FRONTEND_ERROR(validate_sdpa_support_surface_for_implementation(context, implementation));

    return {error_code_t::OK, ""};
}

inline error_t
SDPA_attributes::validate_sdpa_support_surface_for_implementation(const detail::Context& context,
                                                                  AttentionImplementation_t impl) const {
    switch (impl) {
        case AttentionImplementation_t::AUTO:
            // This function should not be called with AUTO.
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                true,
                error_code_t::INVALID_VALUE,
                "Can't call validate_sdpa_support_surface_for_implementation with impl=AUTO");
            break;
        case AttentionImplementation_t::COMPOSITE:
            // Composite implementation already supports all of the features.
            break;
        case AttentionImplementation_t::UNIFIED: {
            auto cudnn_ver_error =
                error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Unified SDPA node requires cuDNN 9.13.0"};
#if (CUDNN_VERSION >= 91300)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91300, cudnn_ver_error);

            for (const auto& [key, value] : inputs) {
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    key != input_names::Q && key != input_names::K && key != input_names::V &&
                        key != input_names::Attn_scale && value != nullptr,
                    error_code_t::GRAPH_NOT_SUPPORTED,
                    "Unified SDPA node doesn't yet support inputs other than Q, K, V and Attn_scale");
            }

            for (const auto& [key, value] : outputs) {
                RETURN_CUDNN_FRONTEND_ERROR_IF(key != output_names::O && key != output_names::Stats && value != nullptr,
                                               error_code_t::GRAPH_NOT_SUPPORTED,
                                               "Unified SDPA node doesn't yet support outputs other than O and Stats");
            }

            RETURN_CUDNN_FRONTEND_ERROR_IF(
                alibi_mask, error_code_t::GRAPH_NOT_SUPPORTED, "Unified SDPA node doesn't yet support alibi mask");

            RETURN_CUDNN_FRONTEND_ERROR_IF(
                padding_mask, error_code_t::GRAPH_NOT_SUPPORTED, "Unified SDPA node doesn't yet support padding mask");

            RETURN_CUDNN_FRONTEND_ERROR_IF(left_bound.has_value() || right_bound.has_value(),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node doesn't yet support left bound or right bound");

            RETURN_CUDNN_FRONTEND_ERROR_IF(diagonal_alignment != DiagonalAlignment_t::TOP_LEFT,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node doesn't yet support diagonal alignment");

            RETURN_CUDNN_FRONTEND_ERROR_IF(dropout_probability.has_value(),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node doesn't yet support dropout");

            RETURN_CUDNN_FRONTEND_ERROR_IF(max_seq_len_kv.has_value(),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node doesn't yet support max sequence length");

            RETURN_CUDNN_FRONTEND_ERROR_IF(attention_score_modifier != nullptr,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node doesn't yet support attention score modifier");

            RETURN_CUDNN_FRONTEND_ERROR_IF(mma_core_mode != DataType_t::HALF,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node doesn't yet support a data type other than fp16");

            int64_t s_q = inputs.at(SDPA_attributes::input_names::Q)->get_dim()[2];
            RETURN_CUDNN_FRONTEND_ERROR_IF(s_q == 1,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node doesn't yet support decode only mode, i.e. s_q == 1");

            RETURN_CUDNN_FRONTEND_ERROR_IF(
                (compute_data_type != DataType_t::NOT_SET && compute_data_type != DataType_t::FLOAT) ||
                    context.get_compute_data_type() != DataType_t::FLOAT,
                error_code_t::GRAPH_NOT_SUPPORTED,
                "Unified SDPA node doesn't yet support compute data type other than float");
#else
            CUDNN_FRONTEND_UNUSED(context);
            return cudnn_ver_error;
#endif
        } break;
    }

    return {error_code_t::OK, ""};
}

}  // namespace cudnn_frontend::graph