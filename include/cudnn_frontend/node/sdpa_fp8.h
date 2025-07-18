#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "matmul_fp8.h"
#include "pointwise.h"
#include "reduction.h"
#include "softmax.h"

namespace cudnn_frontend::graph {

class SDPAFP8Node : public NodeCRTP<SDPAFP8Node> {
    using input_names  = SDPA_fp8_attributes::input_names;
    using output_names = SDPA_fp8_attributes::output_names;

    std::shared_ptr<Tensor_attributes> rng_output;

   public:
    SDPA_fp8_attributes attributes;

    SDPAFP8Node(SDPA_fp8_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating SDPAFP8Node " << attributes.name << "...");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90100,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 forward operation is only supported starting cudnn 9.1.0. Please "
                                       "consider upgrading your current version.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() == 91000,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 forward operation is not supported on cudnn 9.10.0. Please "
                                       "consider upgrading your current version.");

        cudaDeviceProp prop;
        int device;
        CHECK_CUDA_ERROR(detail::cuda_get_device(&device));
        CHECK_CUDA_ERROR(detail::cuda_get_device_properties(&prop, device));
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            prop.major < 9,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Hopper architecture and newer. Please "
            "consider using a newer architecture.");

        // check that Q, K, V, O tensors has been assigned
        // check that dim and strides has been assigned and last stride is 1
#define CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(port, port_map)                                                       \
    {                                                                                                           \
        std::shared_ptr<Tensor_attributes> tensor_ptr = port_map.at(port);                                      \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_dim().size() != 4,                                       \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                         \
                                       "The dim for " + std::string(#port) + " is invalid");                    \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_stride().size() != 4,                                    \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                         \
                                       "The stride for " + std::string(#port) + " is invalid");                 \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                         \
            tensor_ptr->get_stride()[3] != 1,                                                                   \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                  \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + \
                std::string(#port));                                                                            \
    }

        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Q, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::K, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::V, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::O, attributes.outputs);

        // validate options for generate_stats and stats tensor
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.generate_stats.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "generate_stats attribute not set");

        if (attributes.generate_stats.value() == true) {
            CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::Stats);
        }

#undef CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE

        // validate backend limitations for the operation
        int64_t s_q  = attributes.inputs.at(input_names::Q)->get_dim()[2];
        int64_t s_kv = attributes.inputs.at(input_names::K)->get_dim()[2];
        int64_t h_q  = attributes.inputs.at(input_names::Q)->get_dim()[1];
        int64_t h_k  = attributes.inputs.at(input_names::K)->get_dim()[1];
        int64_t h_v  = attributes.inputs.at(input_names::V)->get_dim()[1];
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];

        // bool const is_ragged = attributes.inputs.at(input_names::Q)->get_ragged_offset() ||
        //                        attributes.inputs.at(input_names::K)->get_ragged_offset() ||
        //                        attributes.inputs.at(input_names::V)->get_ragged_offset() ||
        //                        attributes.outputs.at(output_names::O)->get_ragged_offset();

        auto const& bias_mask = attributes.inputs.find(input_names::Bias);
        bool const is_bias    = (bias_mask != attributes.inputs.end() && bias_mask->second != nullptr);

        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        bool const is_dropout        = attributes.dropout_probability.has_value();
        // bool const is_dropout        = attributes.dropout_probability.has_value() || is_dropout_custom;

        // validation TODO:
        //    - validate stats has valid dims

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
        RETURN_CUDNN_FRONTEND_ERROR_IF((h_q % h_k != 0) || (h_q % h_v != 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For group-query attention, number of heads for key and query must be a factor "
                                       "of number of heads for query");

        // validate options for attn_scale
        auto const& attn_scale    = attributes.inputs.find(input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        // validate options for bias mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bias mask data type cannot be boolean");

        // validate options for padding mask
        auto const& seq_len_q     = attributes.inputs.find(input_names::SEQ_LEN_Q);
        bool const has_seq_len_q  = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);
        auto const& seq_len_kv    = attributes.inputs.find(input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        // validate options for dropout mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.dropout_probability.has_value() && is_dropout_custom,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        // Validate options for causal_mask_bottom_right
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask_bottom_right && detail::get_backend_version() < 90700,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.7.0, bottom right causal masking is not supported.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.causal_mask_bottom_right && prop.major < 10,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Blackwell architecture and newer. Please "
            "consider using a newer architecture.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask && attributes.causal_mask_bottom_right,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask and causal mask cannot be both enabled");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.causal_mask_bottom_right && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask does not support s_q > s_kv. Please virtually slice "
                                       "the Q tensor and pass it as s_q == s_kv");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.causal_mask_bottom_right && (is_bias || is_dropout),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Bottom right causal mask is only supported with is_bias=False, is_dropout=False.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.causal_mask_bottom_right && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv multiple of 64");

        // validate that datatype is set for the graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        if (attributes.generate_stats.value() == true) {
            auto stats     = attributes.outputs.at(output_names::Stats);
            auto stats_dim = stats->get_dim();

            if (stats_dim.empty()) {
                // Fill properties of virtual tensors
                auto const& p_dim = attributes.inputs[input_names::Q]->get_dim();
                auto b            = p_dim[0];
                auto h            = p_dim[1];
                auto s_q          = p_dim[2];
                stats->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});
            }
        }
        return {error_code_t::OK, ""};
    }
    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferrencing properties for Scaled_dot_product_flash_attention node  "
                                << attributes.name << "...");

        // DO NOT REMOVE
        // input data type is needed for:
        // - aType of bmm2
        attributes.fill_from_context(context);

        // Gather dim to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h            = q_dim[1];
        auto s_q          = q_dim[2];
        auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        auto s_kv         = k_dim[2];

        // cuDNN frontend API attention requires Q, K, V where
        // Q = {b, h_q, s_q, d_qk}
        // K = {b, h_k, s_kv, d_qk}
        // V = {b, h_v, s_kv, d_v}
        // but cuDNN backend API attention requires Q, KT, V
        // Q = {b, h_q, s_q, d_qk}
        // KT = {b, h_k, d_qk, s_kv}
        // V = {b, h_v, s_kv, d_v}
        // So the code below maps the K->KT
        std::vector<int64_t> temp_vec;

        temp_vec = attributes.inputs[input_names::K]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_dim(temp_vec);

        temp_vec = attributes.inputs[input_names::K]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_stride(temp_vec);

        // This tensor tracks the main chain of data flow
        std::shared_ptr<Tensor_attributes> last_output;
        auto mul_attributes = Pointwise_attributes().set_mode(PointwiseMode_t::MUL);

        //// Q * K
        auto bmm1_attributes = Matmul_attributes()
                                   .set_name("bmm1")
                                   .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                   .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]);

        if (attributes.padding_mask) {
            bmm1_attributes.set_padding(0.0);
        }
        last_output = matmul(attributes.inputs[input_names::Q], attributes.inputs[input_names::K], bmm1_attributes);

        //// Optional Attn scale
        // In case user provided a scalar value, do a fused scalar.
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // If attn scale present, add a pointwise mul node
        if (auto attn_scale_it = attributes.inputs.find(input_names::Attn_scale);
            attn_scale_it != attributes.inputs.end()) {
            mul_attributes.set_name("attn_scale");
            auto const& attn_scale_output = pointwise(last_output, attn_scale_it->second, mul_attributes);
            last_output                   = attn_scale_output;
        }

        //// Descales
        // Descale Q
        mul_attributes.set_name("descale_q");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_Q), mul_attributes);

        // Descale K
        mul_attributes.set_name("descale_k");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_K), mul_attributes);

        // Optional bias
        if (auto bias_it = attributes.inputs.find(input_names::Bias); bias_it != attributes.inputs.end()) {
            auto add_attributes     = Pointwise_attributes().set_name("bias").set_mode(PointwiseMode_t::ADD);
            auto const& bias_output = pointwise(last_output, bias_it->second, add_attributes);
            last_output             = bias_output;
        }

        if (attributes.padding_mask) {
            auto row_index_attributes = Pointwise_attributes()
                                            .set_name("gen_row_index")
                                            .set_mode(PointwiseMode_t::GEN_INDEX)
                                            .set_axis(2)
                                            .set_compute_data_type(DataType_t::INT32);
            auto const& row_index_output = pointwise(last_output, row_index_attributes);
            row_index_output->set_data_type(DataType_t::INT32);

            auto col_index_attributes = Pointwise_attributes()
                                            .set_name("gen_col_index")
                                            .set_mode(PointwiseMode_t::GEN_INDEX)
                                            .set_axis(3)
                                            .set_compute_data_type(DataType_t::INT32);
            auto const& col_index_output = pointwise(last_output, col_index_attributes);
            col_index_output->set_data_type(DataType_t::INT32);

            auto row_less_seq_q_attributes = Pointwise_attributes()
                                                 .set_name("row_less_seq_q")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::INT32);
            auto const& row_less_seq_q_output =
                pointwise(row_index_output, attributes.inputs[input_names::SEQ_LEN_Q], row_less_seq_q_attributes);
            row_less_seq_q_output->set_data_type(DataType_t::INT32);

            auto col_less_seq_kv_attributes = Pointwise_attributes()
                                                  .set_name("col_less_seq_kv")
                                                  .set_mode(PointwiseMode_t::CMP_LT)
                                                  .set_compute_data_type(DataType_t::INT32);
            auto const& col_less_seq_kv_output =
                pointwise(col_index_output, attributes.inputs[input_names::SEQ_LEN_KV], col_less_seq_kv_attributes);
            col_less_seq_kv_output->set_data_type(DataType_t::INT32);

            auto logical_and_attributes = Pointwise_attributes()
                                              .set_name("logical_and")
                                              .set_mode(PointwiseMode_t::LOGICAL_AND)
                                              .set_compute_data_type(DataType_t::BOOLEAN);
            auto const& logical_and_output =
                pointwise(row_less_seq_q_output, col_less_seq_kv_output, logical_and_attributes);
            logical_and_output->set_data_type(DataType_t::BOOLEAN);

            // Lower attributes to binary select attributes
            // Use a smaller value of neg infinity so that the softmax stats for rows that are fully padded dont
            // go towards NaNs/Infs when multipled by the numerous scale/descale
            auto negative_inf_padding =
                std::make_shared<Tensor_attributes>(attn::score_modifiers::get_negative_inf_value());

            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);
            auto const& padding_mask_output =
                pointwise(last_output, negative_inf_padding, logical_and_output, binary_select_attributes);
            last_output = padding_mask_output;
        }

        //// Optional causal or bottom-right causal masking
        if (attributes.causal_mask || attributes.causal_mask_bottom_right) {
            auto row_index_attributes =
                Pointwise_attributes().set_name("gen_row_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            std::shared_ptr<Tensor_attributes> row_index_output = pointwise(last_output, row_index_attributes);
            row_index_output->set_data_type(DataType_t::INT32);

            auto col_index_attributes =
                Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            auto const& col_index_output = pointwise(last_output, col_index_attributes);
            col_index_output->set_data_type(DataType_t::INT32);

            if (attributes.causal_mask_bottom_right) {
                if (attributes.inputs[input_names::SEQ_LEN_KV]) {
                    row_index_output = pointwise(row_index_output,
                                                 attributes.inputs[input_names::SEQ_LEN_KV],
                                                 Pointwise_attributes()
                                                     .set_name("row_idx_add_skv")
                                                     .set_mode(PointwiseMode_t::ADD)
                                                     .set_compute_data_type(DataType_t::INT32));
                } else {
                    row_index_output = pointwise(row_index_output,
                                                 std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_kv)),
                                                 Pointwise_attributes()
                                                     .set_name("row_idx_add_skv")
                                                     .set_mode(PointwiseMode_t::ADD)
                                                     .set_compute_data_type(DataType_t::INT32));
                }
                row_index_output->set_data_type(DataType_t::INT32);

                if (attributes.inputs[input_names::SEQ_LEN_Q]) {
                    row_index_output = pointwise(row_index_output,
                                                 attributes.inputs[input_names::SEQ_LEN_Q],
                                                 Pointwise_attributes()
                                                     .set_name("row_idx_add_sq_sub_sq")
                                                     .set_mode(PointwiseMode_t::SUB)
                                                     .set_compute_data_type(DataType_t::INT32));
                } else {
                    row_index_output = pointwise(row_index_output,
                                                 std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_q)),
                                                 Pointwise_attributes()
                                                     .set_name("row_idx_add_sq_sub_sq")
                                                     .set_mode(PointwiseMode_t::SUB)
                                                     .set_compute_data_type(DataType_t::INT32));
                }
                row_index_output->set_data_type(DataType_t::INT32);
            }

            auto greater_than_attributes = Pointwise_attributes()
                                               .set_name("row_greater_than_col")
                                               .set_mode(PointwiseMode_t::CMP_GE)
                                               .set_compute_data_type(DataType_t::BOOLEAN);
            auto const& row_greater_than_col_output =
                pointwise(row_index_output, col_index_output, greater_than_attributes);
            row_greater_than_col_output->set_data_type(DataType_t::BOOLEAN);

            // Lower attributes to binary select attributes
            auto negative_inf_causal =
                std::make_shared<Tensor_attributes>(attn::score_modifiers::get_negative_inf_value());

            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);
            auto const& causal_mask_output =
                pointwise(last_output, negative_inf_causal, row_greater_than_col_output, binary_select_attributes);
            last_output = causal_mask_output;
        }

        //// Softmax
        // softmax output, S, is always virtual.
        auto softmax_output = std::make_shared<Tensor_attributes>();
        softmax_output->set_is_virtual(true);

        // Create virtual stats if inference step otherwise output.Stats should be provided by user.
        auto softmax_stats = attributes.outputs[output_names::Stats];
        if (attributes.generate_stats.value() == false) {
            softmax_stats = std::make_shared<Tensor_attributes>();
            softmax_stats->set_is_virtual(true);
        }

        auto softmax_attributes =
            Softmax_attributes().set_name("softmax").has_stats(true).has_M_Zinv(false);  // As this is flash attention
        // Special non-functional-style call. Needed because output already created and provided to user.
        softmax(last_output, softmax_attributes, softmax_output, softmax_stats);
        last_output = softmax_output;

        // Two cases for training: dropout present or not
        bool dropout_present         = false;
        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        if (attributes.dropout_probability.has_value()) {
            dropout_present = true;
            // Special case: Skip dropout when 0.0 probability.
            if (attributes.dropout_probability.value() == 0.0) {
                dropout_present = false;
            }
        } else if (is_dropout_custom) {
            dropout_present = true;
        }

        if (dropout_present) {
            if (is_dropout_custom) {
                auto dropout_scale_attributes =
                    Pointwise_attributes().set_name("dropout_scale_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_scale_output =
                    pointwise(last_output, attributes.inputs[input_names::Dropout_scale], dropout_scale_attributes);

                auto mask_attributes =
                    Pointwise_attributes().set_name("dropout_mask_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_mask_output =
                    pointwise(dropout_scale_output, dropout_mask->second, mask_attributes);
                last_output = dropout_mask_output;
            } else {
                rng_output = rng(attributes.inputs[input_names::Seed],
                                 attributes.inputs[input_names::Offset],
                                 Rng_attributes()
                                     .set_name("rng")
                                     .set_distribution(RngDistribution_t::BERNOULLI)
                                     .set_bernoulli_probability(1.0 - attributes.dropout_probability.value()));
                rng_output
                    // Hard coding dim and strides as rng output can no inputs to infer it from.
                    ->set_dim({b, h, s_q, s_kv})
                    .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

                auto mask_attributes =
                    Pointwise_attributes().set_name("dropout_mask_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_mask_output = pointwise(last_output, rng_output, mask_attributes);
                last_output                     = dropout_mask_output;

                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> dropout_scale = nullptr;

                float dropout_scale_value = (1.0f / (1.0f - attributes.dropout_probability.value()));
                dropout_scale             = std::make_shared<Tensor_attributes>(dropout_scale_value);

                auto dropout_scale_attributes =
                    Pointwise_attributes().set_name("dropout_scale").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_scale_output = pointwise(last_output, dropout_scale, dropout_scale_attributes);
                last_output                      = dropout_scale_output;
            }
        }

        // Amax S
        auto amax_attributes = Reduction_attributes().set_name("amax_s").set_mode(ReductionMode_t::AMAX);
        // Special non-functional-style call. Needed because output already created and provided to user.
        reduction(last_output, amax_attributes, attributes.outputs.at(output_names::Amax_S));

        // Scale S
        mul_attributes.set_name("scale_s");
        last_output = pointwise(last_output, attributes.inputs.at(input_names::Scale_S), mul_attributes);
        last_output->set_data_type(attributes.inputs.at(input_names::Q)->get_data_type());

        // Lower attributes to bmm2 attributes
        // Requirement by cudnn backend to take in bmm2 aType as i/o type.
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        //// S * V
        auto bmm2_attributes = Matmul_fp8_attributes()
                                   .set_name("bmm2")
                                   .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                   .set_k_override(attributes.inputs[input_names::SEQ_LEN_KV]);
        // Special non-functional-style call. Needed because output already created and provided to user.
        matmul_fp8(last_output,
                   attributes.inputs.at(input_names::V),
                   attributes.inputs.at(input_names::Descale_S),
                   attributes.inputs.at(input_names::Descale_V),
                   attributes.inputs.at(input_names::Scale_O),
                   bmm2_attributes,
                   attributes.outputs.at(output_names::O),
                   attributes.outputs.at(output_names::Amax_O));

        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
#define CUDNN_FE_VALIDATE_STRIDE(port, port_map)                                                                \
    {                                                                                                           \
        auto const& t = port_map.find(port);                                                                    \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                         \
            t->second->get_stride().back() != 1,                                                                \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                  \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + \
                std::string(#port));                                                                            \
    }

        CUDNN_FE_VALIDATE_STRIDE(output_names::O, attributes.outputs);

#undef CUDNN_FE_VALIDATE_STRIDE

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "SDPA_FP8_FWD"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph